"""## Executive summary (read this first)

Prepare fresh mounted inputs for the frozen 25-scenario GLS hybrid. Official
monthly targets receive a 25-percent whole-path mixture. Eligible daily targets
receive only bounded common shifts. Other cards use GLS without model calls.
Reuse research prompts, parsing, empirical draws and forecast transformations.
Do not read cached forecasts or outcomes; an optional callable enables offline
replay validation. This module alone is not a published submission.
"""
import tomllib

import numpy as np
import pandas as pd

import cordp
from engine import _full_series
from forecast_instructions import instructed_paths
from forecast_schema import forecast_schema
from history_context_ablation import with_history_length
from pipelines.house_scenario_client import HouseScenarioClient, HouseUnavailable
from pipelines.joint_residual_transport import common_path_mixture
from pipelines.numerical_combinations import combine, conditional_common


def routing(card, cells, info, log_columns, mode):
    if mode not in ('monthly_only', 'monthly_daily'):
        raise ValueError('Unknown frozen hybrid mode')
    if card['targets']['target_frequency'] == 'monthly':
        return 'monthly_raw_mixture25'
    assets = {c[0] for c in cells}
    if (mode == 'monthly_daily' and not info['monthly'] and not log_columns and len(cells) > 1
            and (len(assets) == 1 or all(a.startswith('UST') for a in assets))):
        return 'daily_bounded_common25'
    return 'gls_only'


def forecast(unit, *, call=None, mode='monthly_daily', news=True, row_seed=0):
    has_spec = (unit / 'forecast_spec.json').is_file()
    if has_spec:
        x, cells, info, context, docs = cordp.prepare(unit, 0)
        context = with_history_length(context, unit, 60)
    else:
        # The public minimal exemplar has no forecast_spec. Retain the numerical
        # forecast instead of inventing monthly periods or failing before export.
        x, cells, info = cordp.baseline(unit, 0)
        context, docs = None, []
    card = tomllib.loads((unit / 'card.toml').read_text())
    log_columns, positive_columns = [], []
    for asset in {c[0] for c in cells}:
        _, stem = _full_series(unit, asset, pd.Timestamp(info['asof']))
        if info['target'] == 'level' and 'fx' in stem.lower():
            log_columns.extend(i for i, cell in enumerate(cells) if cell[0] == asset)
        if info['target'] == 'level' and ('fx' in stem.lower() or stem == 'em_transfer_early'):
            positive_columns.extend(i for i, cell in enumerate(cells) if cell[0] == asset)
    gls, calibration = conditional_common(x, .75, positive_columns)
    route = routing(card, cells, info, log_columns, mode)
    if not has_spec:
        route = 'gls_only'
    audit = {'route': route, 'mode': mode, 'news': news, 'row_seed': row_seed,
             'gls': calibration, 'calls': 0, 'model_response_errors': [],
             'local_replay': call is not None, 'house_schema_support_verified': False}
    if not has_spec:
        audit['input_fallback'] = 'Missing forecast_spec; retained numerical forecast'
    if route == 'gls_only':
        return gls, cells, info, audit
    client = None
    try:
        if call is None:
            response_format = {'type': 'json_schema', 'json_schema': {'name': 'forecast',
                               'strict': True, 'schema': forecast_schema(context)}}
            client = HouseScenarioClient(response_format)
            call = client
        def counted(role, system, prompt, temperature, seed):
            if audit['calls'] >= 25:
                raise RuntimeError('Scenario request limit reached')
            audit['calls'] += 1
            return call(role, system, prompt, temperature, seed)
        proposal, generation = instructed_paths(context, docs, counted, news=news, variant='scenario')
    except HouseUnavailable as exc:
        audit['service_fallback'] = str(exc)
        if client is not None:
            audit['calls'] = client.calls
            audit['request_ledger'] = client.ledger
        return gls, cells, info, audit
    audit['model_response_errors'] = generation['errors']
    if client is not None:
        audit['request_ledger'] = client.ledger
    if positive_columns and np.any(proposal[:, positive_columns] <= 0):
        audit['support_fallback'] = 'Nonpositive FX proposal; retain GLS'
        return gls, cells, info, audit
    if route == 'monthly_raw_mixture25':
        result = combine([gls, proposal], [.75, .25], 'joint_mixture', seed=row_seed)
    else:
        result = common_path_mixture(gls, proposal, .25, seed=row_seed, radius=.5)
    return result, cells, info, audit
