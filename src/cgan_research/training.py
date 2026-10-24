"""Bounded reference training, fixed-noise generation, and exact local resume."""
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint, write_json
from .models import Generator, Discriminator


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    device: str = 'cuda'
    batch_size: int = 2000
    max_epochs: int = 10000
    max_steps: int = 200
    max_seconds: float = 600.0
    sampling: str = 'fixed'
    g_lr: float = 0.0001
    d_lr: float = 0.0002
    betas: tuple = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    checkpoint_interval: int = 50
    validation_interval: int = 100
    evaluation_seed: int = 42
    samples_per_condition: int = 2000
    cpu_threads: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'betas', tuple(self.betas))
        for name in ('batch_size', 'max_epochs', 'max_steps', 'checkpoint_interval', 'samples_per_condition', 'cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.validation_interval) is not int or self.validation_interval < 0:
            raise ValueError('validation_interval must be nonnegative')
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('max_seconds must be finite and positive')
        if self.sampling not in ('fixed', 'shuffle', 'balanced'):
            raise ValueError('sampling must be fixed, shuffle, or balanced')
        if self.device not in ('cuda', 'cpu'):
            raise ValueError('device must be cuda or cpu')
        if not all(math.isfinite(v) and v > 0 for v in (self.g_lr, self.d_lr, self.epsilon)):
            raise ValueError('learning rates and epsilon must be finite and positive')
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('invalid Adam configuration')


@dataclass
class GeneratedBatch:
    loads: np.ndarray
    labels: np.ndarray
    units: str
    data_fingerprint: str
    seed: int
    checkpoint_fingerprint: str
    representation: str = 'physical'
    model: str = 'reference_cgan'
    training_seed: int | None = None
    split_fingerprint: str = ''
    condition_mapping: dict | None = None


def config_dict(config):
    return json.loads(json.dumps(asdict(config)))


def capture_rng():
    n = np.random.get_state()
    return {'python_rng': random.getstate(),
            'numpy_rng': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python_rng'])
    n = state['numpy_rng']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def environment(device):
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[2], text=True))
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    result = {'python': platform.python_version(), 'torch': str(torch.__version__),
              'numpy': np.__version__, 'cuda_build': torch.version.cuda,
              'device': device, 'git_sha': sha, 'git_dirty': dirty}
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(source.name.encode()); digest.update(source.read_bytes())
    result['source_sha256'] = digest.hexdigest()
    result['packages'] = {name:version(name) for name in ('torch','numpy','pandas','scipy','matplotlib','PyYAML')}
    if device == 'cuda' and torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(), total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    return result


def generate_from_model(generator, conditions, samples_per_condition, seed, normalizer):
    if samples_per_condition < 1 or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError('unique supported conditions and positive sample count required')
    if any(type(c) is not int or c < 0 or c >= generator.num_conditions for c in conditions):
        raise ValueError('unsupported condition')
    labels = np.repeat(conditions, samples_per_condition).astype(np.int64)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(len(labels), 24, generator=rng)
    device = next(generator.parameters()).device
    was_training = generator.training
    generator.eval()
    try:
        with torch.no_grad():
            chunks = [generator(z.to(device), torch.as_tensor(y, device=device)).cpu().numpy()
                      for z, y in zip(noise.split(4096), np.array_split(labels, np.arange(4096, len(labels), 4096)))]
        model_loads = np.concatenate(chunks).astype(np.float64)
        physical = model_loads * normalizer['scale'] + normalizer['min']
        if not np.isfinite(physical).all():
            raise FloatingPointError('Nonfinite generated profiles')
        return physical, labels
    finally:
        generator.train(was_training)


def generate(checkpoint, conditions, samples_per_condition=2000, seed=42):
    state = load_checkpoint(checkpoint)
    # Constructing nn.Linear consumes the global RNG; fork it as well as the noise.
    with torch.random.fork_rng(devices=[]):
        generator = Generator(len(state['condition_mapping']))
    generator.load_state_dict(state['generator'])
    loads, labels = generate_from_model(generator, conditions, samples_per_condition, seed, state['normalizer'])
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return GeneratedBatch(loads, labels, state['units'], state['data_fingerprint'], seed, digest,
        model='reference_cgan' if state['configuration']['sampling']=='fixed' else 'cgan_'+state['configuration']['sampling'],
        training_seed=state['configuration']['seed'],split_fingerprint=state['split_fingerprint'],condition_mapping=state['condition_mapping'])


def configure_cuda():
    value = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if value not in (':4096:8', ':16:8'):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before CUDA initialization')


def train(prepared, config, run_dir, resume=None, validation_callback=None, stop_after_steps=None):
    """Train up to the declared cumulative step/time budget. stop_after_steps simulates a pause."""
    if isinstance(config, dict):
        config = TrainConfig(**config)
    configure_cuda()
    directory = Path(run_dir)
    cfg = config_dict(config)
    expected = {key: prepared.manifest[key] for key in ('data_fingerprint', 'split_fingerprint')}
    expected['configuration'] = cfg
    if resume is not None:
        state = load_checkpoint(resume, expected)
        if Path(resume).resolve().parent != directory.resolve() or not directory.is_dir():
            raise ValueError('Resume must use the original run directory')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        state = None
        write_json(directory/'config.json', cfg)
        write_json(directory/'provenance.json', {'prepared': prepared.manifest, 'environment': environment(config.device)})
    started = time.monotonic()
    elapsed_before = state.get('elapsed_seconds', 0.0) if state else 0.0
    try:
        if config.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; no CPU fallback is allowed')
        if config.validation_interval and validation_callback is None:
            from .evaluation import make_validation_callback
            validation_callback = make_validation_callback(prepared, config.samples_per_condition, config.evaluation_seed)
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if config.device == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats()
        torch.use_deterministic_algorithms(True)
        nclasses = len(prepared.manifest['condition_mapping'])
        g, d = Generator(nclasses).to(config.device), Discriminator(nclasses).to(config.device)
        go = torch.optim.Adam(g.parameters(), lr=config.g_lr, betas=config.betas, eps=config.epsilon, weight_decay=config.weight_decay)
        do = torch.optim.Adam(d.parameters(), lr=config.d_lr, betas=config.betas, eps=config.epsilon, weight_decay=config.weight_decay)
        indices = np.asarray(prepared.splits['train'])
        x = torch.from_numpy(np.asarray(prepared.normalized[indices], dtype=np.float32))
        y = torch.from_numpy(np.asarray(prepared.labels[indices], dtype=np.int64))
        if not len(x) or not torch.isfinite(x).all() or y.min() < 0:
            raise ValueError('Invalid training data')
        if state is None:
            state = dict(version=1, **expected, condition_mapping=prepared.manifest['condition_mapping'],
                         normalizer=prepared.manifest['normalizer'], units=prepared.manifest['units'],
                         step=0, epoch=0, cursor=0, order=torch.empty(0,dtype=torch.long), history=[], epoch_history=[],
                         epoch_g_sum=0., epoch_d_sum=0., epoch_samples=0, best_score=None, best_step=None,
                         next_validation_step=config.validation_interval, validation_history=[],
                         evaluation_seed=config.evaluation_seed, selected_checkpoint=None)
        else:
            g.load_state_dict(state['generator']); d.load_state_dict(state['discriminator'])
            go.load_state_dict(state['g_optimizer']); do.load_state_dict(state['d_optimizer'])
            restore_rng(state)
            if state['selected_checkpoint']:
                selected = load_checkpoint(directory/state['selected_checkpoint'], expected)
                if selected['step'] != state['best_step']:
                    raise ValueError('Selected checkpoint step mismatch')
                save_checkpoint(directory/'best.pt', selected)
        loss_fn = nn.BCEWithLogitsLoss()

        def checkpoint(selected=False):
            state.update(generator=g.state_dict(), discriminator=d.state_dict(), g_optimizer=go.state_dict(), d_optimizer=do.state_dict(),
                         elapsed_seconds=elapsed_before + time.monotonic()-started, **capture_rng())
            # Publish an immutable selected artifact before last.pt references it.
            # best.pt is a recoverable convenience copy; last.pt is authoritative.
            if selected:
                save_checkpoint(directory/state['selected_checkpoint'], state)
            save_checkpoint(directory/'last.pt', state)
            if selected:
                save_checkpoint(directory/'best.pt', state)
            pd.DataFrame(state['history']).to_csv(directory/'history.csv', index=False)
            pd.DataFrame(state['epoch_history']).to_csv(directory/'epochs.csv', index=False)
            write_json(directory/'validation.json', state['validation_history'])

        if resume is None:
            checkpoint()
        reason = 'step_budget'
        while state['step'] < config.max_steps and state['epoch'] < config.max_epochs:
            if elapsed_before + time.monotonic()-started >= config.max_seconds:
                reason = 'time_budget'; break
            if stop_after_steps is not None and state['step'] >= stop_after_steps:
                reason = 'paused'; break
            if not len(state['order']):
                if config.sampling == 'fixed':
                    order = np.arange(len(x))
                elif config.sampling == 'shuffle':
                    order = np.random.permutation(len(x))
                else:
                    counts = np.bincount(y.numpy(), minlength=nclasses)
                    weights = 1.0/counts[y.numpy()]
                    order = np.random.choice(len(x), len(x), replace=True, p=weights/weights.sum())
                state['order'] = torch.as_tensor(order, dtype=torch.long)
            batch = state['order'][state['cursor']:state['cursor']+config.batch_size]
            real, labels = x[batch].to(config.device), y[batch].to(config.device)
            do.zero_grad(set_to_none=True)
            noise = torch.randn(len(batch),24,device=config.device)
            fake = g(noise, labels)
            real_logits, fake_logits = d(real, labels), d(fake.detach(), labels)
            dloss = (loss_fn(real_logits, torch.ones_like(real_logits)) + loss_fn(fake_logits, torch.zeros_like(fake_logits)))/2
            if not torch.isfinite(dloss): raise FloatingPointError('Nonfinite discriminator loss')
            dloss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in d.parameters()):
                raise FloatingPointError('Nonfinite discriminator gradient')
            do.step()
            for p in d.parameters(): p.requires_grad_(False)
            go.zero_grad(set_to_none=True)
            logits = d(fake, labels)
            gloss = loss_fn(logits, torch.ones_like(logits))
            if not torch.isfinite(gloss): raise FloatingPointError('Nonfinite generator loss')
            gloss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in g.parameters()):
                raise FloatingPointError('Nonfinite generator gradient')
            go.step()
            for p in d.parameters(): p.requires_grad_(True)
            if any(not torch.isfinite(p).all() for p in list(g.parameters())+list(d.parameters())):
                raise FloatingPointError('Nonfinite model parameter')
            state['step'] += 1; state['cursor'] += len(batch)
            gl, dl = float(gloss.detach()), float(dloss.detach())
            state['history'].append({'step':state['step'], 'epoch':state['epoch'], 'samples':len(batch), 'g_loss':gl, 'd_loss':dl})
            state['epoch_g_sum'] += gl*len(batch); state['epoch_d_sum'] += dl*len(batch); state['epoch_samples'] += len(batch)
            if state['cursor'] == len(x):
                state['epoch'] += 1
                state['epoch_history'].append({'epoch':state['epoch'],'step':state['step'],'samples':state['epoch_samples'],
                    'g_loss':state['epoch_g_sum']/state['epoch_samples'],'d_loss':state['epoch_d_sum']/state['epoch_samples']})
                state.update(cursor=0, order=torch.empty(0,dtype=torch.long), epoch_g_sum=0.,epoch_d_sum=0.,epoch_samples=0)
            selected = False
            if config.validation_interval and state['step'] == state['next_validation_step']:
                rng = capture_rng()
                try:
                    result = validation_callback(g, state['step'])
                finally:
                    restore_rng(rng)
                if result.get('metadata',{}).get('split') != 'validation':
                    raise ValueError('Checkpoint selection requires validation-only results')
                score = result.get('score')
                if result['available'] and (score is None or not math.isfinite(score)):
                    raise FloatingPointError('Invalid validation score')
                state['validation_history'].append({'step':state['step'], **result})
                state['next_validation_step'] += config.validation_interval
                if result['available'] and (state['best_score'] is None or score < state['best_score']):
                    state.update(best_score=score, best_step=state['step'], selected_checkpoint=f"best-step-{state['step']:08d}.pt")
                    selected = True
            if selected or state['step'] % config.checkpoint_interval == 0:
                checkpoint(selected=selected)
        if state['epoch'] >= config.max_epochs: reason = 'epoch_budget'
        checkpoint()
        result = {'status':'paused' if reason == 'paused' else 'completed','stop_reason':reason,
                  'steps':state['step'],'completed_epochs':state['epoch'], 'elapsed_seconds':state['elapsed_seconds'],
                  'steps_per_second':state['step']/max(state['elapsed_seconds'],1e-12),
                  'best_step':state['best_step'],'best_validation_score':state['best_score'],
                  'device':config.device, 'peak_allocated_bytes':torch.cuda.max_memory_allocated() if config.device=='cuda' else None}
        write_json(directory/'status.json', result)
        return result
    except BaseException as exc:
        # Last atomic checkpoint is left intact; replay after it is deterministic.
        write_json(directory/'status.json', {'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            'error_type':type(exc).__name__, 'error':str(exc), 'device':config.device,
            'steps':state.get('step',0) if state else 0, 'elapsed_seconds':elapsed_before+time.monotonic()-started})
        raise
