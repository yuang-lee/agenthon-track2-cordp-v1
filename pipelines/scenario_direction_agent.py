"""## Executive summary (read this first)

Run numerical location controls with House directions on eligible Treasury grids
on freshly mounted task inputs. Verify the code manifest and input cutoff before
calling the model. Export the three official contract files, with truthful route,
request counts and fallbacks. Never include endpoint credentials or cached answers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pipelines'), str(ROOT / 'scripts')]
from engine import write
from pipelines.scenario_direction_hybrid import forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', nargs='?', choices=['forecast'], default='forecast')
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--asof', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.is_absolute() or args.out.name != 'forecast.parquet':
        parser.error('--out must be an absolute path ending in forecast.parquet')
    unit = next((p for p in (args.panels, args.panels.parent) if (p / 'card.toml').is_file()), None)
    if unit is None or args.text.resolve() != (unit / 'text').resolve():
        parser.error('panels and text must belong to the same mounted task')
    card = tomllib.loads((unit / 'card.toml').read_text())
    cutoff = str(card['provenance']['data_cutoff'])[:10]
    if args.asof[:10] != cutoff:
        parser.error('--asof disagrees with the task data_cutoff')
    freeze = json.loads((ROOT / 'freeze.json').read_text())
    for name, digest in freeze['files_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen package file changed: ' + name)
    settings = freeze['settings']
    if settings['row_seed'] != 0 or type(settings['news']) is not bool:
        raise ValueError('Unexpected frozen scenario settings')
    samples, cells, info, audit = forecast(unit, mode=settings['mode'], news=settings['news'], row_seed=0)
    write(samples, cells, info, args.out)
    fallback = audit.get('service_fallback') or audit.get('support_fallback')
    actual_route = audit['actual_route']
    meta_path = args.out.parent / 'forecast_meta.json'
    meta = json.loads(meta_path.read_text())
    meta['rationale']['method'] = ('Frozen numerical baseline with GLS common shrinkage0.75; '
                                  'House scenario route: ' + actual_route)
    meta['house_requests'] = audit['calls']
    meta['reasoning_applied'] = audit['reasoning_applied']
    meta['reasoning_skipped_reason'] = (fallback or audit.get('input_fallback') or
                                       ('numerical-only route' if not audit['reasoning_applied'] else ''))
    meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    details = {'asof': cutoff, 'n_draws': len(samples), 'baseline_seed': 0, 'mixture_row_seed': 0,
        'settings': settings, 'requested_route': audit['route'], 'actual_route': actual_route,
        'house_requests': audit['calls'], 'max_requests': 25, 'max_output_tokens_per_request': 3000,
        'thinking': False, 'temperature': 1., 'invalid_scenario_count': len(audit['model_response_errors']),
        'invalid_scenario_policy': 'raw-history last level or zero cumulative return, preserving the frozen experiment',
        'service_failure_policy': 'retain numerical common-center for rates, GLS for monthly; stop without retry', 'fallback': fallback,
        'input_fallback': audit.get('input_fallback'),
        'request_ledger': audit.get('request_ledger', []), 'cells': cells,
        'baseline_rule_notes': info.get('rule_notes', []),
        'orientation': audit.get('orientation')}
    (args.out.parent / 'forecast_rationale.md').write_text(
        '## Executive summary (read this first)\n\n'
        'The numerical forecast uses GLS calibration. Daily Treasury grids receive a bounded mixture '
        'toward the numerical common center.25 House scenarios select whole-row relative orientation, '
        'preserving all within-row absolute differences. Single-asset daily FX uses the same bounded '
        'mixture toward the last observation with zero model requests. Monthly targets retain a25percent '
        'whole-scenario mixture; other tasks retain GLS. Model outputs are proposals, not calibrated '
        'event probabilities. Actual routing and numerical fallbacks follow.\n\n'
        + json.dumps(details, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {len(samples)} joint draws; route={actual_route}; House requests={audit["calls"]}')


if __name__ == '__main__':
    main()
