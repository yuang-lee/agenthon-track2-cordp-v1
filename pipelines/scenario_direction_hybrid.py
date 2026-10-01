"""## Executive summary (read this first)

Separate numerical location control from House scenario direction. Daily
Treasury grids use a fixed numerical common-center mixture; House proposals
select relative orientation only. Single-asset daily FX grids use a numerical
last-observation mixture with no model calls. Monthly tasks retain the frozen
whole-scenario mixture. Other inputs retain GLS. No outcomes or cached forecasts
are read. Failed model requests preserve the route's numerical forecast.
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
    if mode != 'monthly_daily' or info['monthly'] or len(cells)<2 or info['target']!='level':
        return 'gls_only'
    assets = {a for a, _ in cells}
    if not log_columns and all(a.startswith('UST') for a in assets):
        return 'daily_rates_numeric_center_direction'
    if len(assets)==1 and len(log_columns)==len(cells):
        return 'daily_fx_persistence'
    return 'gls_only'


def forecast(unit, *, call=None, mode='monthly_daily', news=True, row_seed=0):
    has_spec = (unit/'forecast_spec.json').is_file()
    if has_spec:
        x,cells,info,context,docs = cordp.prepare(unit,0)
        context = with_history_length(context,unit,60)
    else:
        x,cells,info = cordp.baseline(unit,0)
        context,docs = None,[]
    card = tomllib.loads((unit/'card.toml').read_text())
    log_columns,positive_columns,last = [],[],{}
    for asset in {a for a,_ in cells}:
        series,stem = _full_series(unit,asset,pd.Timestamp(info['asof']))
        last[asset] = float(series.iloc[-1])
        if info['target']=='level' and 'fx' in stem.lower():
            log_columns.extend(i for i,c in enumerate(cells) if c[0]==asset)
        if info['target']=='level' and ('fx' in stem.lower() or stem=='em_transfer_early'):
            positive_columns.extend(i for i,c in enumerate(cells) if c[0]==asset)
    gls,calibration = conditional_common(x,.75,positive_columns)
    route = routing(card,cells,info,log_columns,mode) if has_spec else 'gls_only'
    numerical = gls.copy()
    if route=='daily_rates_numeric_center_direction':
        center = np.broadcast_to(gls.mean(axis=0),gls.shape).copy()
        numerical = common_path_mixture(gls,center,.25,seed=row_seed,radius=.5)
    elif route=='daily_fx_persistence':
        center = np.broadcast_to([last[a] for a,_ in cells],gls.shape).copy()
        numerical = common_path_mixture(gls,center,.25,seed=row_seed,radius=.5)
    audit = dict(route=route,actual_route=route,mode=mode,news=news,row_seed=row_seed,
        gls=calibration,calls=0,reasoning_applied=False,model_response_errors=[],
        local_replay=call is not None,house_schema_support_verified=False)
    if not has_spec:
        audit['input_fallback']='Missing forecast_spec; retained numerical forecast'
    if not np.isfinite(numerical).all() or (positive_columns and np.any(numerical[:,positive_columns]<=0)):
        numerical = gls.copy()
        audit.update(support_fallback='Numerical correction violates support; retain GLS',actual_route='gls_only')
    if route in ('gls_only','daily_fx_persistence'):
        return numerical,cells,info,audit
    client = None
    try:
        if call is None:
            response_format={'type':'json_schema','json_schema':{'name':'forecast',
                             'strict':True,'schema':forecast_schema(context)}}
            client=HouseScenarioClient(response_format)
            call=client
        def counted(role,system,prompt,temperature,seed):
            if audit['calls']>=25:
                raise RuntimeError('Scenario request limit reached')
            audit['calls']+=1
            return call(role,system,prompt,temperature,seed)
        proposal,generation = instructed_paths(context,docs,counted,news=news,variant='scenario')
    except HouseUnavailable as exc:
        audit.update(service_fallback=str(exc),actual_route=(
            'daily_rates_numeric_center' if route=='daily_rates_numeric_center_direction' else 'gls_only'))
        if client is not None:
            audit.update(calls=client.calls,request_ledger=client.ledger)
        return numerical,cells,info,audit
    audit['model_response_errors']=generation['errors']
    if client is not None:
        audit['request_ledger']=client.ledger
    if route=='monthly_raw_mixture25':
        result=combine([gls,proposal],[.75,.25],'joint_mixture',seed=row_seed)
    else:
        relative=gls-gls.mean(axis=1,keepdims=True)
        desired=proposal-proposal.mean(axis=1,keepdims=True)
        flip=(relative*desired).sum(axis=1)<0
        result=numerical.copy()
        result[flip]=2*result[flip].mean(axis=1,keepdims=True)-result[flip]
        audit['orientation']=dict(active_flipped_rows=int(flip.sum()),orientation_fraction=1.,
            common_fraction=.25,common_radius=.5,location_source='numerical column means')
    if not np.isfinite(result).all() or (positive_columns and np.any(result[:,positive_columns]<=0)):
        audit.update(support_fallback='Scenario transform violates support; retain numerical forecast',
            actual_route='daily_rates_numeric_center' if route=='daily_rates_numeric_center_direction' else 'gls_only')
        return numerical,cells,info,audit
    audit['reasoning_applied']=True
    return result,cells,info,audit
