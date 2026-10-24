"""Headless, evidence-preserving figures for the approved calendar baseline.

Explicit limits are shared unchanged across runs. Without them, limits are
inferred from this evaluation and recorded; those exports are not a fixed-scale
cross-model comparison. Histograms use raw empirical fractions and retain tails.
"""
import json
from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

PALETTE = {'real': '#e68613', 'generated': '#008b8b'}
SEASONS = ('winter', 'spring', 'summer', 'autumn')
DAY_TYPES = ('weekday', 'weekend')
CONDITIONS = tuple(f'{season}|{day}' for day in DAY_TYPES for season in SEASONS)
METRICS = ('mean_difference', 'std_difference', 'kl', 'frechet')


def _limits(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Display limits require [minimum, maximum]')
    bounds = np.asarray(value, dtype=float)
    if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('Display limits must be finite and increasing')
    return bounds.tolist()


def _validate_config(config):
    if not isinstance(config, dict) or set(config) - {'profile_ylim', 'heatmap_limits', 'histogram_xlim'}:
        raise ValueError('Figure config supports profile_ylim, heatmap_limits, histogram_xlim only')
    if 'profile_ylim' in config:
        _limits(config['profile_ylim'])
    for key, names in (('heatmap_limits', METRICS), ('histogram_xlim', FEATURE_NAMES[:2])):
        mapping = config.get(key, {})
        if not isinstance(mapping, dict) or set(mapping) - set(names):
            raise ValueError(f'Unsupported {key} names')
        for value in mapping.values():
            if isinstance(value, dict) and key == 'heatmap_limits':
                if set(value) - set(FEATURE_NAMES):
                    raise ValueError('Unknown heatmap feature')
                for bounds in value.values():
                    _limits(bounds)
            else:
                _limits(value)


def _inferred_limits(values, signed=False):
    finite = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(finite):
        return [-1., 1.] if signed else [0., 1.]
    if signed:
        extent = max(float(np.max(np.abs(finite))), 1e-12)
        return [-extent, extent]
    low, high = float(finite.min()), float(finite.max())
    margin = (high - low) * .05 if high != low else max(abs(low) * .05, .5)
    return [low - margin, high + margin]


def render_figures(evaluation_dir, output_dir, figure_config=None):
    """Export all calendar figure families and exact source tables, without overwrite.

    Heatmap limits may be metric-wide pairs or per-feature pairs, for example
    ``{'heatmap_limits': {'kl': [0, 1], 'frechet': {'near_peak': [0, 4]}}}``.
    Mixed feature units always receive separate heatmap panels/colorbars.
    """
    config = {} if figure_config is None else figure_config
    _validate_config(config)
    source = Path(evaluation_dir) / 'evaluation.json'
    result = json.loads(source.read_text())
    metadata = result['metadata']
    for key in ('split', 'model', 'seed', 'units', 'version', 'condition_mapping'):
        if key not in metadata:
            raise ValueError(f'Evaluation metadata missing {key}')
    if set(metadata['condition_mapping']) - set(CONDITIONS):
        raise ValueError('Figures require approved season|weekday/weekend conditions')
    for key in ('rows', 'mean_profiles', 'histograms'):
        if not isinstance(result.get(key), list):
            raise ValueError(f'Evaluation requires {key} rows')
        for row in result[key]:
            if row['condition'] not in CONDITIONS:
                raise ValueError('Unsupported calendar condition in figure source')
            for name in ('value', 'mean', 'fraction'):
                if row.get(name) is not None and not np.isfinite(row[name]):
                    raise ValueError('Nonfinite figure source value')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tables = output / 'source_tables'
    tables.mkdir()
    shutil.copyfile(source, tables / 'evaluation.json')
    for key, value in result.items():
        if isinstance(value, list):
            pd.DataFrame(value).to_csv(tables / f'{key}.csv', index=False)
    synthetic = bool(metadata.get('provenance', {}).get('synthetic', False))
    profiles = result['mean_profiles']
    counts = {condition: {source: max((r['n'] for r in profiles if r['condition'] == condition and r['source'] == source), default=0)
                          for source in PALETTE} for condition in CONDITIONS}
    unavailable = [c for c in CONDITIONS if not all(counts[c].values())]
    manifest = {'metadata': metadata, 'synthetic': synthetic, 'config': config,
                'palette': dict(PALETTE), 'unavailable_conditions': unavailable,
                'counts': counts, 'warnings': [], 'figures': [],
                'source_tables': 'source_tables', 'dpi': 300,
                'scale_policy': 'Explicit configured limits are fixed; other limits are evaluation-derived and recorded.',
                'histogram_estimator': 'raw empirical bin fractions; training-defined edges; tails reported separately'}
    caption = (('SYNTHETIC DEMONSTRATION | ' if synthetic else '') +
               f"{metadata['split']} | {metadata['model']} | training seed {metadata['seed']} | generation seed {metadata.get('generation_seed', 'unrecorded')} | {metadata['version']} | {metadata['units']}\n"
               'Counts in panels/manifest; n_real < 50 exploratory; missing values unavailable. Differences: generated minus real.')

    def warn(family, kind, **detail):
        record = {'family': family, 'kind': kind, **detail}
        manifest['warnings'].append(record)
        warnings.warn(f'{family}: {kind}; values outside display or tails retained in source tables: {detail}', UserWarning, stacklevel=2)

    def saturation(family, values, bounds, **detail):
        outside = sum(v is not None and (v < bounds[0] or v > bounds[1]) for v in values)
        if outside:
            warn(family, 'display_saturation', count=int(outside), limits=bounds, **detail)

    def save(fig, family, title, scales=None):
        fig.suptitle(title, fontsize=14, y=.99)
        fig.text(.02, .015, caption, fontsize=7, va='bottom')
        fig.tight_layout(rect=(0, .095, 1, .95))
        files = {}
        try:
            for extension in ('pdf', 'svg', 'png'):
                relative = f'{family}.{extension}'
                fig.savefig(output / relative, dpi=300)
                files[extension] = relative
        finally:
            plt.close(fig)
        manifest['figures'].append({'family': family, 'files': files, 'scales': scales or {}})

    def panel_title(condition):
        n = counts[condition]
        return f"{condition.replace('|', ' / ')}\nreal n={n['real']}; generated n={n['generated']}"

    def unavailable_panel(ax):
        ax.text(.5, .5, 'Unavailable', transform=ax.transAxes, ha='center', va='center', color='#666666')
        ax.set_facecolor('#eeeeee')

    with plt.rc_context({'svg.fonttype': 'none', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}):
        for family in ('workflow', 'architecture'):
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.set_axis_off()
            if family == 'workflow':
                boxes = ['Audit physical\n24-hour profiles', 'Freeze meter/group\nsplits + train scales',
                         'Calendar cGAN\ntrain / resume', 'Validation selection\nthen frozen test', 'Tables + figures\nretain provenance']
                title = 'Original research workflow'
            else:
                boxes = ['24-D normal noise\n+ K calendar one-hot', 'Generator\n24+K → 64 → 128\n→ 64 → 24; linear',
                         'Model-space output\nReal inputs use same\ntraining transform', 'Real / fake +\nK calendar one-hot',
                         'Discriminator\n24+K → 64 → 128\n→ 256 → 1 logit']
                title = 'Original reference cGAN architecture'
            for i, label in enumerate(boxes):
                x = .01 + i * .2
                ax.add_patch(FancyBboxPatch((x, .38), .17, .35, boxstyle='round,pad=.01',
                                          transform=ax.transAxes, facecolor='#e5f2f1', edgecolor='#008b8b'))
                ax.text(x + .085, .555, label, ha='center', va='center', transform=ax.transAxes, fontsize=9)
                if i:
                    ax.annotate('', xy=(x, .555), xytext=(x - .03, .555), xycoords='axes fraction', arrowprops={'arrowstyle': '->'})
            if family == 'architecture':
                ax.text(.5, .18, 'Hidden LeakyReLU 0.2 | D dropout 0.4 | BCEWithLogitsLoss | one D and one G update per batch\nExported generated profiles: inverse training normalization to physical units',
                        transform=ax.transAxes, ha='center', fontsize=9)
            save(fig, family, title)

        ylim = _limits(config['profile_ylim']) if 'profile_ylim' in config else _inferred_limits([r['mean'] for r in profiles])
        saturation('mean_profiles', [r['mean'] for r in profiles], ylim)
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
        for ax, condition in zip(axes.flat, CONDITIONS):
            ax.set_title(panel_title(condition))
            for source, color in PALETTE.items():
                rows = sorted((r for r in profiles if r['condition'] == condition and r['source'] == source), key=lambda r: r['hour'])
                if any(r['mean'] is not None for r in rows):
                    ax.plot([r['hour'] for r in rows], [np.nan if r['mean'] is None else r['mean'] for r in rows], color=color, label=source,
                            gid=f"mean-profile-{condition.replace('|', '-')}-{source}")
            if condition in unavailable:
                unavailable_panel(ax)
            ax.set(xlim=(0, 23), ylim=ylim, xlabel='Hour', ylabel=metadata['units'])
            if ax.lines:
                ax.legend(fontsize=7)
        save(fig, 'mean_profiles', 'Seasonal mean daily profiles — all meter groups', {'ylim': ylim})

        for family, metrics in (('mean_std', METRICS[:2]), ('kl', ('kl',)), ('frechet', ('frechet',))):
            fig, axes = plt.subplots(len(metrics), len(FEATURE_NAMES), figsize=(18, 4.6 * len(metrics)), squeeze=False)
            scales = {}
            for row_index, metric in enumerate(metrics):
                for col, feature in enumerate(FEATURE_NAMES):
                    ax = axes[row_index, col]
                    cells = {r['condition']: r for r in result['rows'] if r['feature'] == feature and r['metric'] == metric}
                    values = [cells.get(c, {}).get('value') for c in CONDITIONS]
                    configured = config.get('heatmap_limits', {}).get(metric)
                    if isinstance(configured, dict):
                        configured = configured.get(feature)
                    bounds = _limits(configured) if configured is not None else _inferred_limits(values, metric.endswith('difference'))
                    key = f'{metric}/{feature}' if len(metrics) > 1 else feature
                    scales[key] = bounds
                    saturation(family, values, bounds, feature=feature, metric=metric)
                    matrix = np.asarray([np.nan if v is None else v for v in values]).reshape(2, 4)
                    cmap = plt.get_cmap('coolwarm' if metric.endswith('difference') else 'viridis').copy()
                    cmap.set_bad('#dddddd')
                    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=bounds[0], vmax=bounds[1], cmap=cmap, aspect='auto')
                    for i in range(2):
                        for j in range(4):
                            value = matrix[i, j]
                            label = 'Unavailable' if np.isnan(value) else f'{value:.3g}'
                            ax.text(j, i, label, ha='center', va='center', fontsize=6, rotation=45 if np.isnan(value) else 0,
                                    bbox={'facecolor': 'white', 'alpha': .65, 'edgecolor': 'none', 'pad': .5})
                    unit = next((r['units'] for r in cells.values()), ('hours' if col >= 2 else metadata['units']))
                    ax.set_title(f'{feature}\n{metric}', fontsize=9)
                    ax.set_xticks(range(4), SEASONS, rotation=45, ha='right')
                    ax.set_yticks(range(2), DAY_TYPES)
                    fig.colorbar(image, ax=ax, shrink=.7, label=unit)
            save(fig, family, {'mean_std': 'Feature mean and sample-standard-deviation differences', 'kl': 'KL(real || generated), natural logarithms',
                               'frechet': 'Per-feature Gaussian Fréchet distance (not image FID)'}[family], scales)

        for feature in FEATURE_NAMES[:2]:
            family = f'{feature}_histogram'
            histogram = [r for r in result['histograms'] if r['feature'] == feature]
            edges = result.get('reference', {}).get('edges', {}).get(feature, [])
            finite_edges = [e for r in histogram for e in (r['left'], r['right']) if e is not None]
            default = [min(finite_edges), max(finite_edges)] if finite_edges and min(finite_edges) < max(finite_edges) else [0, 1]
            bounds = _limits(config.get('histogram_xlim', {}).get(feature, [edges[0], edges[-1]] if len(edges) >= 2 else default))
            fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
            for ax, condition in zip(axes.flat, CONDITIONS):
                ax.set_title(panel_title(condition))
                notes = []
                for source, color in PALETTE.items():
                    rows = sorted((r for r in histogram if r['condition'] == condition and r['source'] == source), key=lambda r: r['bin'])
                    interior = [r for r in rows if r['left'] is not None and r['right'] is not None]
                    if interior:
                        ax.stairs([r['fraction'] for r in interior], [interior[0]['left']] + [r['right'] for r in interior], color=color, label=source)
                    tails = sum(r['count'] for r in rows if r['left'] is None or r['right'] is None)
                    if tails:
                        warn(family, 'histogram_tails', condition=condition, source=source, count=tails)
                    notes.append(f'{source} tails n={tails}')
                    hidden = sum(r['count'] for r in interior if r['left'] < bounds[0] or r['right'] > bounds[1])
                    if hidden:
                        warn(family, 'display_saturation', condition=condition, source=source, count=hidden, limits=bounds)
                if condition in unavailable:
                    unavailable_panel(ax)
                else:
                    ax.text(.02, .98, '\n'.join(notes), transform=ax.transAxes, va='top', fontsize=6)
                ax.set(xlim=bounds, ylim=(0, 1), xlabel=metadata['units'], ylabel='Empirical fraction / bin')
                if ax.patches:
                    ax.legend(fontsize=7, loc='upper right')
            save(fig, family, f'{feature.replace("_", " ").title()} distributions — all calendar classes; tails reported separately',
                 {'xlim': bounds, 'ylim': [0, 1]})
    encoded_manifest = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    (output / 'figure_manifest.json').write_text(encoded_manifest)
    (output / 'manifest.json').write_text(encoded_manifest)
    return manifest
