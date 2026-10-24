"""Validation-only selection and physical-unit distribution diagnostics."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from .checkpoint import write_json
from .features import FEATURE_NAMES, METRIC_VERSION, daily_features
from .metrics import compare_distributions, fit_metric_reference, fit_reference_from_prepared, histogram_probabilities
from .training import GeneratedBatch, generate_from_model

PUBLIC_PROVENANCE = ('synthetic','timezone','timestamp_semantics','units','quantity','interval_convention','grouping_semantics','ev_detection_semantics','heat_pump_documentation_caveat','pv_caveat','weather_used','source_archive_sha256','source_archive_bytes','source_integrity','expected_release_sha256','expected_release_bytes','published_release_sha256','published_release_bytes','published_release_checksum_matches','calendar')

METRICS = ('mean_difference','abs_mean_difference','std_difference','abs_std_difference','kl','js','wasserstein','frechet')


def empirical_baseline(prepared, samples_per_condition=2000, seed=42, training_seed=None):
    if samples_per_condition < 1:
        raise ValueError('Positive sample count required')
    rng = np.random.default_rng(seed)
    train = prepared.splits['train']
    samples, labels = [], []
    for label in range(len(prepared.manifest['condition_mapping'])):
        pool = train[prepared.labels[train] == label]
        if not len(pool):
            raise ValueError('Condition has no training records')
        samples.append(prepared.loads[rng.choice(pool,samples_per_condition,replace=True)])
        labels.extend([label]*samples_per_condition)
    return GeneratedBatch(np.concatenate(samples),np.asarray(labels,dtype=np.int64),prepared.manifest['units'],
        prepared.manifest['data_fingerprint'],seed,'training_resampling',model='empirical_resampling',training_seed=training_seed,
        split_fingerprint=prepared.manifest['split_fingerprint'],condition_mapping=prepared.manifest['condition_mapping'])


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['feature'],row['metric'])].append(row)
    result = []
    for (feature,metric), cells in groups.items():
        good = [c for c in cells if c['value'] is not None]
        result.append({'feature':feature,'metric':metric,
            'macro':float(np.mean([c['value'] for c in good])) if good else None,
            'real_count_weighted':float(np.average([c['value'] for c in good],weights=[c['n_real'] for c in good])) if good else None,
            'available_cells':len(good),'unavailable_cells':len(cells)-len(good),
            'excluded_real_count':sum(c['n_real'] for c in cells if c['value'] is None)})
    return result


def _autocorrelation(x):
    centered = x-x.mean(axis=1,keepdims=True)
    denominator = np.sum(centered**2,axis=1)
    valid = denominator > 0
    values = [float(np.mean(np.sum(centered[valid,:-lag]*centered[valid,lag:],axis=1)/denominator[valid])) if valid.any() else None for lag in range(1,7)]
    return values,int((~valid).sum())


def _diversity(x, pairs=1000, seed=42):
    if len(x) < 2:
        return None
    # Canonical row order makes seeded pair estimation permutation invariant.
    ordered = x[np.lexsort(x.T[::-1])]
    rng = np.random.default_rng(seed)
    a = rng.integers(len(x),size=pairs)
    b = rng.integers(len(x)-1,size=pairs)
    b += b >= a
    return float(np.sqrt(np.mean((ordered[a]-ordered[b])**2,axis=1)).mean())


def profile_diagnostics(real, generated, training):
    r,g,t = (np.asarray(x,dtype=np.float64) for x in (real,generated,training))
    if any(x.ndim!=2 or x.shape[1]!=24 or not np.isfinite(x).all() for x in (r,g,t)):
        raise ValueError('Expected finite physical profiles')
    if not len(r) or not len(g):
        return {'reason':'empty_distribution','hourly_mean_rmse':None}
    ra,rc = _autocorrelation(r); ga,gc = _autocorrelation(g)
    nearest = cKDTree(t).query(g,k=1,workers=1)[0]/np.sqrt(24) if len(t) else None
    return {'reason':None,'hourly_mean_rmse':float(np.sqrt(np.mean((r.mean(axis=0)-g.mean(axis=0))**2))),
            'daily_energy_wasserstein':float(wasserstein_distance(r.sum(axis=1),g.sum(axis=1))),
            'ramp_wasserstein':float(wasserstein_distance(np.diff(r,axis=1).ravel(),np.diff(g,axis=1).ravel())),
            'real_autocorrelation':ra,'generated_autocorrelation':ga,
            'real_constant_profiles':rc,'generated_constant_profiles':gc,
            'real_diversity_rms':_diversity(r),'generated_diversity_rms':_diversity(g),
            'diversity_pairs':1000,'diversity_seed':42,
            'nearest_training_rms_mean':float(nearest.mean()) if nearest is not None else None,
            'nearest_training_rms_min':float(nearest.min()) if nearest is not None else None,
            'exact_training_match_rate':float((nearest==0).mean()) if nearest is not None else None}


def validation_score(evaluation, reference):
    if evaluation['metadata']['split'] != 'validation':
        raise ValueError('Selection requires validation data')
    kept = {name:reference['scale'][i] for i,name in enumerate(FEATURE_NAMES) if not reference['constant'][i]}
    by_condition = defaultdict(list)
    for row in evaluation['rows']:
        if row['metric']=='wasserstein' and row['feature'] in kept and row['value'] is not None:
            by_condition[row['condition']].append(row['value']/kept[row['feature']])
    scores = [float(np.mean(values)) for values in by_condition.values() if len(values)==len(kept)]
    return {'score':float(np.mean(scores)) if scores else None, 'available':bool(scores),
            'metadata':{'split':'validation','constant_features_excluded':[n for n in FEATURE_NAMES if n not in kept],
                        'available_conditions':len(scores),'unsupported_real_count':evaluation['metadata'].get('unsupported_real_count',0),
                        'reason':None if scores else 'no_nonconstant_supported_validation_features'}}


def make_validation_callback(prepared, samples_per_condition=2000, seed=42):
    reference = fit_reference_from_prepared(prepared)
    idx = prepared.splits['validation']
    real_features = daily_features(prepared.loads[idx])
    labels = prepared.labels[idx]
    mapping = prepared.manifest['condition_mapping']
    def callback(generator, step):
        generated, generated_labels = generate_from_model(generator,list(range(len(mapping))),samples_per_condition,seed,prepared.manifest['normalizer'])
        generated_features = daily_features(generated)
        rows = []
        for name,label in mapping.items():
            r,g = real_features[labels==label],generated_features[generated_labels==label]
            for i,feature in enumerate(FEATURE_NAMES):
                rows.append({'condition':name,'feature':feature,'metric':'wasserstein',
                             'value':float(wasserstein_distance(r[:,i],g[:,i])) if len(r) else None})
        return validation_score({'rows':rows,'metadata':{'split':'validation','unsupported_real_count':int((labels<0).sum())}},reference)
    return callback


def evaluate(prepared, generated, reference, split='validation', reference_replicates=100):
    if split not in ('validation','test'):
        raise ValueError('Evaluation partition must be validation or test')
    if generated.representation != 'physical' or generated.units != prepared.manifest['units']:
        raise ValueError('Evaluation requires matching physical units')
    if generated.split_fingerprint != prepared.manifest['split_fingerprint'] or generated.condition_mapping != prepared.manifest['condition_mapping']:
        raise ValueError('Generated split/condition mapping fingerprint mismatch')
    if generated.data_fingerprint != prepared.manifest['data_fingerprint'] or reference.get('data_fingerprint') != prepared.manifest['data_fingerprint'] or reference.get('split_fingerprint') != prepared.manifest['split_fingerprint']:
        raise ValueError('Evaluation data/reference fingerprint mismatch')
    if reference_replicates < 0:
        raise ValueError('reference_replicates cannot be negative')
    mapping = prepared.manifest['condition_mapping']
    experiment_seed = generated.training_seed if generated.training_seed is not None else generated.seed
    if generated.labels.shape != (len(generated.loads),) or generated.labels.dtype.kind not in 'iu' or not np.isin(generated.labels,list(mapping.values())).all():
        raise ValueError('Unsupported generated conditions')
    idx,train = prepared.splits[split],prepared.splits['train']
    real = prepared.loads[idx]
    rf,gf = daily_features(real),daily_features(generated.loads)
    train_features = daily_features(prepared.loads[train])
    sensitivities = {bins:fit_metric_reference(train_features,bins) for bins in (15,30,60)}
    result = {'metadata':{'split':split,'model':generated.model,'seed':experiment_seed,'units':generated.units,
              'version':METRIC_VERSION,'data_fingerprint':generated.data_fingerprint,'split_fingerprint':prepared.manifest['split_fingerprint'],
              'generation_seed':generated.seed,'checkpoint_fingerprint':generated.checkpoint_fingerprint,'condition_mapping':mapping,
              'unsupported_real_count':int((prepared.labels[idx]<0).sum()),'n_real_total':len(idx),
              'n_generated_total':len(generated.loads),'provenance':{k:v for k,v in prepared.manifest.get('provenance',{}).items() if k in PUBLIC_PROVENANCE},
              'reference_replicates':reference_replicates,'reference_seed':42,
              'finite_sample_reference':'independent with-replacement real resamples; not a performance bound',
              'feature_difference_direction':'generated minus real','privacy_claim':False},
              'rows':[],'profile_metrics':[],'mean_profiles':[],'histograms':[],'sensitivity':[],'real_reference':[],
              'reference':reference}
    rng = np.random.default_rng(42)
    for name,label in mapping.items():
        mask,gm = prepared.labels[idx]==label,generated.labels==label
        r,g,fr,fg = real[mask],generated.loads[gm],rf[mask],gf[gm]
        comparison = compare_distributions(fr,fg,reference)
        for i,feature in enumerate(FEATURE_NAMES):
            cell = comparison['features'][feature]
            for metric in METRICS:
                value = cell[metric]
                unit = 'nats' if metric in ('kl','js') else ('hours' if i>=2 else generated.units)
                if metric=='frechet': unit += '^2'
                result['rows'].append({'condition':name,'condition_id':label,'feature':feature,'metric':metric,
                    'value':value,'reason':(cell['reason'] or cell['variance_reason']) if value is None else None,
                    'n_real':len(r),'n_generated':len(g),'exploratory':len(r)<50,'split':split,
                    'model':generated.model,'seed':experiment_seed,'units':unit,'version':METRIC_VERSION})
            if i<2:
                edges = reference['edges'][feature]
                for source,values in (('real',fr[:,i]),('generated',fg[:,i])):
                    if len(values):
                        # Raw empirical proportions for figures, no pseudocount alteration.
                        e=np.array(edges); codes=np.searchsorted(e,values,side='right'); codes[values==e[-1]]=len(e)-1
                        counts=np.bincount(codes,minlength=len(e)+1)
                        for b,count in enumerate(counts):
                            result['histograms'].append({'condition':name,'feature':feature,'source':source,'bin':b,
                                'left':float(e[b-1]) if b else None,'right':float(e[b]) if b<len(e) else None,
                                'count':int(count),'fraction':float(count/len(values))})
        diagnostic = profile_diagnostics(r,g,prepared.loads[train])
        result['profile_metrics'].append({'condition':name,'n_real':len(r),'n_generated':len(g),
            'joint_feature_frechet':comparison['joint_frechet'],'joint_reason':comparison['joint_reason'],**diagnostic})
        for source,values in (('real',r),('generated',g)):
            for hour in range(24):
                result['mean_profiles'].append({'condition':name,'source':source,'hour':hour,
                    'mean':float(values[:,hour].mean()) if len(values) else None,'n':len(values)})
        for bins,ref in sensitivities.items():
            comp = compare_distributions(fr,fg,ref)
            for feature in FEATURE_NAMES[:2]:
                result['sensitivity'].append({'condition':name,'feature':feature,'bins':bins,'kl':comp['features'][feature]['kl']})
        if len(fr) and len(fg):
            active = ~np.array(reference['constant'])
            scale = np.array(reference['scale'])
            for replicate in range(reference_replicates):
                a,b = fr[rng.integers(len(fr),size=len(fg))],fr[rng.integers(len(fr),size=len(fg))]
                distances = [wasserstein_distance(a[:,i],b[:,i])/scale[i] for i in range(5) if active[i]]
                result['real_reference'].append({'condition':name,'replicate':replicate,'sample_count':len(fg),
                    'scaled_feature_wasserstein':float(np.mean(distances)) if distances else None})
    unsupported = prepared.labels[idx] < 0
    if unsupported.any():
        names = np.asarray(getattr(prepared, 'conditions', np.full(len(prepared.loads), 'unsupported_training_condition')))[idx]
        for name in sorted(set(names[unsupported])):
            count = int((unsupported & (names==name)).sum())
            for i,feature in enumerate(FEATURE_NAMES):
                for metric in METRICS:
                    unit = 'nats' if metric in ('kl','js') else ('hours' if i>=2 else generated.units)
                    if metric=='frechet': unit += '^2'
                    result['rows'].append({'condition':str(name),'condition_id':-1,'feature':feature,'metric':metric,
                        'value':None,'reason':'condition_absent_from_training','n_real':count,'n_generated':0,
                        'exploratory':count<50,'split':split,'model':generated.model,'seed':experiment_seed,'units':unit,'version':METRIC_VERSION})
    result['summary'] = aggregate_rows(result['rows'])
    # Private grouped feature records are separate from shareable JSON/tables.
    # Never serialize this cache into a public report.
    result['_private'] = {'real_features':rf,'real_labels':prepared.labels[idx],
        'real_groups':np.asarray(getattr(prepared,'household_ids',np.full(len(prepared.loads),'')))[idx],
        'generated_features':gf,'generated_labels':generated.labels}
    return result


def save_evaluation(evaluation, output_dir):
    directory = Path(output_dir); directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'evaluation.json',{k:v for k,v in evaluation.items() if not k.startswith('_')})
    if '_private' in evaluation:
        np.savez_compressed(directory/'private_records.npz',**evaluation['_private'])
    for name in ('rows','summary','mean_profiles','histograms','sensitivity','real_reference'):
        pd.DataFrame(evaluation[name]).to_csv(directory/f'{name}.csv',index=False)


def cluster_bootstrap(values, groups, statistic, resamples=1000, seed=42):
    values,groups = np.asarray(values),np.asarray(groups)
    if not len(values) or len(groups)!=len(values) or resamples<1:
        raise ValueError('Nonempty aligned groups and positive resamples required')
    unique = np.unique(groups)
    if len(unique)<2:
        return {'estimate':float(statistic(values)),'interval_95':None,'n_groups':len(unique),'reason':'fewer_than_two_groups'}
    members = [np.flatnonzero(groups==g) for g in unique]
    rng = np.random.default_rng(seed)
    draws = [float(statistic(values[np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])])) for _ in range(resamples)]
    return {'estimate':float(statistic(values)),'interval_95':np.quantile(draws,[.025,.975]).tolist(),
            'n_groups':len(unique),'resamples':resamples,'seed':seed,'reason':None}


def compare_runs(run_dirs, bootstrap_seed=42, resamples=1000):
    """Paired seed effects and paired meter/household-cluster selection-score intervals."""
    if resamples < 1: raise ValueError('Positive bootstrap replicates required')
    records = [json.loads((Path(d)/'evaluation.json').read_text()) for d in run_dirs]
    if not records:
        raise ValueError('No evaluation records')
    identities={(r['metadata']['data_fingerprint'],r['metadata']['split_fingerprint'],r['metadata']['split'],r['metadata']['version'],r['metadata']['units']) for r in records}
    if len(identities)!=1:
        raise ValueError('Comparisons require shared data, split, units, and metric version')
    base_reference = records[0]['reference']
    base_mapping = records[0]['metadata']['condition_mapping']
    for record in records:
        if record['reference'] != base_reference or record['metadata']['condition_mapping'] != base_mapping:
            raise ValueError('Comparisons require identical training metric references and condition mappings')
    groups=defaultdict(dict)
    for record in records:
        model,seed=record['metadata']['model'],record['metadata']['seed']
        if seed in groups[model]: raise ValueError('Duplicate model/seed evaluation')
        groups[model][seed]=record
    if len(groups)!=2:
        raise ValueError('Exactly two named models required for paired comparison')
    first,second=sorted(groups)
    seeds=sorted(set(groups[first])&set(groups[second]))
    result={'models':[first,second],'paired_seeds':seeds,'unpaired_seeds':{m:sorted(set(g)-set(seeds)) for m,g in groups.items()},
            'effects':[],'household_intervals':None,'limitation':'Separate seed and meter/household-cluster intervals; no joint two-level uncertainty or population representativeness claim.'}
    if not seeds: return result
    for training_seed in seeds:
        a,b = groups[first][training_seed],groups[second][training_seed]
        if a['metadata'].get('generation_seed') != b['metadata'].get('generation_seed'):
            raise ValueError('Paired comparison requires identical generation sampling seeds')
        def counts(record):
            result = {}
            for row in record['rows']:
                value = (row['n_real'],row['n_generated'])
                if row['condition'] in result and result[row['condition']] != value:
                    raise ValueError('Inconsistent sample counts in evaluation')
                result[row['condition']] = value
            return result
        if counts(a) != counts(b):
            raise ValueError('Paired comparison requires equal real/generated sample counts by condition')
    result['household_intervals'] = _paired_cluster_interval(run_dirs, records, first, second, seeds, bootstrap_seed, resamples)
    for i,cell in enumerate(groups[first][seeds[0]]['summary']):
        differences=[]
        for seed in seeds:
            a,b=groups[first][seed]['summary'][i],groups[second][seed]['summary'][i]
            if (a['feature'],a['metric'])!=(b['feature'],b['metric']): raise ValueError('Incompatible metric tables')
            if a['macro'] is not None and b['macro'] is not None: differences.append(b['macro']-a['macro'])
        if differences:
            interval=cluster_bootstrap(np.array(differences),np.arange(len(differences)),np.mean,resamples,bootstrap_seed)
            result['effects'].append({'feature':cell['feature'],'metric':cell['metric'],'direction':f'{second} minus {first}',**interval})
    return result


def _paired_cluster_interval(run_dirs, records, first, second, seeds, seed, resamples):
    caches = {}
    references = {}
    for directory,record in zip(run_dirs,records):
        key=(record['metadata']['model'],record['metadata']['seed'])
        if key[1] not in seeds: continue
        path=Path(directory)/'private_records.npz'
        if not path.exists():
            return {'interval_95':None,'reason':'private_grouped_records_unavailable','n_groups':None}
        with np.load(path,allow_pickle=False) as store:
            caches[key]={k:store[k] for k in store.files}
        references[key]=record['reference']
    base=caches[(first,seeds[0])]
    real,labels,groups=base['real_features'],base['real_labels'],base['real_groups']
    if (groups=='').any():
        return {'interval_95':None,'reason':'group_identifiers_unavailable','n_groups':None}
    for key,cache in caches.items():
        for field in ('real_features','real_labels','real_groups'):
            if not np.array_equal(cache[field],base[field]):
                raise ValueError('Paired cluster evaluation requires identical real grouped records')
        if references[key] != references[(first,seeds[0])]:
            raise ValueError('Paired comparison requires identical training metric references')
    for training_seed in seeds:
        a,b=caches[(first,training_seed)],caches[(second,training_seed)]
        if not np.array_equal(np.bincount(a['generated_labels']),np.bincount(b['generated_labels'])):
            raise ValueError('Paired models require equal generated counts per condition')
    unique=np.unique(groups)
    supported=sorted(set(labels[labels>=0].tolist()))
    active=~np.array(references[(first,seeds[0])]['constant'])
    scale=np.array(references[(first,seeds[0])]['scale'])
    if len(unique)<2 or not active.any():
        return {'interval_95':None,'reason':'insufficient_groups_or_nonconstant_features','n_groups':len(unique)}
    def effect(indices):
        values=[]
        for training_seed in seeds:
            scores={}
            for model in (first,second):
                cache=caches[(model,training_seed)]; by_condition=[]
                for label in supported:
                    r=real[indices][labels[indices]==label]
                    g=cache['generated_features'][cache['generated_labels']==label]
                    if not len(r) or not len(g): return None
                    by_condition.append(np.mean([wasserstein_distance(r[:,i],g[:,i])/scale[i] for i in range(5) if active[i]]))
                if not by_condition: return None
                scores[model]=np.mean(by_condition)
            values.append(scores[second]-scores[first])
        return float(np.mean(values))
    members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(resamples):
        indices=np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])
        value=effect(indices)
        if value is not None: draws.append(value)
    return {'metric':'macro_feature_wasserstein_scaled_by_training_sd','direction':f'{second} minus {first}',
        'estimate':effect(np.arange(len(real))),'interval_95':np.quantile(draws,[.025,.975]).tolist() if draws else None,
        'n_groups':len(unique),'paired_seeds':seeds,'resamples':resamples,'seed':seed,'available_replicates':len(draws),
        'unavailable_replicates':resamples-len(draws),'unsupported_real_count':int((labels<0).sum()),
        'scope':'Whole grouping units resampled jointly across paired models and seeds; meters are household proxies where documented.',
        'reason':None if draws else 'no_complete_supported_condition_replicates'}
