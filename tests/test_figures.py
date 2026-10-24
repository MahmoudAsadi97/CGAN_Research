"""Publication exports must preserve evidence, missing cells, and declared scales."""
import json
import struct
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def evaluation_fixture(directory):
    directory.mkdir()
    condition = 'winter|weekday'
    result = {
        'metadata': {'split': 'validation', 'model': 'reference_cgan', 'seed': 17,
                     'units': 'kWh', 'version': 'corrected-v1',
                     'condition_mapping': {condition: 0}, 'provenance': {'synthetic': True}},
        'rows': [{'condition': condition, 'feature': feature, 'metric': metric,
                  'value': value, 'n_real': 2, 'n_generated': 3, 'exploratory': True,
                  'units': 'nats' if metric == 'kl' else 'kWh'}
                 for feature in ('near_peak', 'near_base')
                 for metric, value in (('mean_difference', 2.5), ('std_difference', None),
                                       ('kl', .5), ('frechet', 7.))],
        'mean_profiles': [{'condition': condition, 'source': source, 'hour': hour,
                           'mean': value, 'n': count}
                          for source, value, count in (('real', -1., 2), ('generated', 3., 3))
                          for hour in range(24)],
        'histograms': [{'condition': condition, 'feature': feature, 'source': source,
                        'bin': b, 'left': left, 'right': right, 'count': count, 'fraction': count / total}
                       for feature in ('near_peak', 'near_base')
                       for source, total, counts in (('real', 2, (1, 1, 0)), ('generated', 3, (0, 1, 2)))
                       for b, (left, right, count) in enumerate(zip((None, 0., 1.), (0., 1., None), counts))],
        'sensitivity': [{'condition': condition, 'feature': 'near_peak', 'bins': 15, 'kl': .4}],
        'reference': {'edges': {'near_peak': [0., 1.], 'near_base': [0., 1.]}}
    }
    (directory / 'evaluation.json').write_text(json.dumps(result))
    return result


def test_headless_exports_preserve_tables_missing_panels_scales_and_tails(tmp_path):
    from cgan_research.figures import render_figures
    source = evaluation_fixture(tmp_path / 'evaluation')
    cfg = {'profile_ylim': [-2, 2], 'heatmap_limits': {'kl': [0, .25]},
           'histogram_xlim': {'near_peak': [0, 1], 'near_base': [0, 1]}}
    with pytest.warns(UserWarning, match='saturat|outside'):
        result = render_figures(tmp_path / 'evaluation', tmp_path / 'figures', cfg)
    assert json.loads(json.dumps(result)) == result
    assert result['synthetic'] is True and result['metadata'] == source['metadata']
    assert result['config'] == cfg
    assert result['palette'] == {'real': '#e68613', 'generated': '#008b8b'}
    assert result['unavailable_conditions'] == ['spring|weekday', 'summer|weekday', 'autumn|weekday',
                                             'winter|weekend', 'spring|weekend', 'summer|weekend', 'autumn|weekend']
    assert any(w['family'] == 'mean_profiles' for w in result['warnings'])
    assert any(w['family'] == 'kl' for w in result['warnings'])
    assert any(w['family'] == 'near_peak_histogram' and w['kind'] == 'histogram_tails' for w in result['warnings'])
    copied = json.loads((tmp_path / 'figures' / 'source_tables' / 'evaluation.json').read_text())
    assert copied == source
    assert (tmp_path / 'figures' / 'source_tables' / 'rows.csv').is_file()
    families = {f['family']: f for f in result['figures']}
    assert set(families) == {'workflow', 'architecture', 'mean_profiles', 'mean_std', 'kl',
                             'frechet', 'near_peak_histogram', 'near_base_histogram'}
    for family in families.values():
        assert set(family['files']) == {'pdf', 'svg', 'png'}
        for extension, relative in family['files'].items():
            data = (tmp_path / 'figures' / relative).read_bytes()
            assert len(data) > 100
            if extension == 'png':
                offset = data.index(b'pHYs')
                x, y, unit = struct.unpack('>IIB', data[offset + 4:offset + 13])
                assert unit == 1 and x * .0254 >= 299.9 and y * .0254 >= 299.9
    svg = (tmp_path / 'figures' / families['mean_profiles']['files']['svg']).read_text()
    assert '#e68613' in svg and '#008b8b' in svg
    assert 'Unavailable' in svg and 'kWh' in svg and 'SYNTHETIC' in svg
    assert 'EV' not in svg
    # Compare actual SVG curve positions: -1 real lies below +3 generated.
    tree = ET.fromstring(svg)
    paths = []
    for source_name in ('real', 'generated'):
        group = tree.find(f".//*[@id='mean-profile-winter-weekday-{source_name}']")
        assert group is not None
        path = group.find('{http://www.w3.org/2000/svg}path')
        paths.append([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', path.attrib['d'])])
    assert paths[0][1] > paths[1][1]
    assert families['mean_profiles']['scales']['ylim'] == [-2., 2.]
    assert families['kl']['scales']['near_peak'] == [0., .25]
    assert families['kl']['scales']['near_base'] == [0., .25]
    assert json.loads((tmp_path / 'figures' / 'figure_manifest.json').read_text()) == result
    assert json.loads((tmp_path / 'figures' / 'manifest.json').read_text()) == result
    with pytest.raises(FileExistsError):
        render_figures(tmp_path / 'evaluation', tmp_path / 'figures', cfg)


@pytest.mark.parametrize('config', [{'profile_ylim': [2, 1]}, {'heatmap_limits': {'kl': [0, float('inf')]}},
                                   {'histogram_xlim': {'near_peak': [0]}}, {'seasons': ['EV scenario']}])
def test_bad_scale_or_unsupported_config_cannot_leave_partial_exports(tmp_path, config):
    from cgan_research.figures import render_figures
    evaluation_fixture(tmp_path / 'evaluation')
    with pytest.raises(ValueError):
        render_figures(tmp_path / 'evaluation', tmp_path / 'figures', config)
    assert not (tmp_path / 'figures').exists()
