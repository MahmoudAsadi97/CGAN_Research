"""Atomic local checkpoints containing tensors and primitive metadata only."""
import json
import os
from pathlib import Path
import tempfile
import torch


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, expected=None):
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Invalid or incomplete checkpoint: {Path(path).name}') from exc
    if not isinstance(state, dict) or state.get('version') != 1:
        raise ValueError('Unsupported checkpoint version')
    for key, value in (expected or {}).items():
        if state.get(key) != value:
            raise ValueError(f'Incompatible checkpoint {key}')
    return state


def write_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
