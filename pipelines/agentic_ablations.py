"""## Executive summary (read this first)

Simple controls isolate median steering, the gate, the reviewer and the supplied
numerical forecast. All use the same dated histories/news and joint-cell parser.
The LLM-only control receives no rule-based forecast or distribution parameters;
its invalid replies use a declared raw-history persistence/zero-return fallback.
No outcome, fitted threshold, Gaussian smoothing or independent cell shuffle is used.
"""
import copy
import json

import numpy as np

import cordp
from cordp_samples import SYSTEM as PATH_SYSTEM, parse_trajectory, empirical_draws
from evidence_transport import (SYSTEM, PROPOSE, REVIEW, transport_context, payload,
                                parse_map, transport_samples)

DIRECT = """Forecast every requested target from the supplied numerical history and dated
documents. No existing model forecast is provided. Use only information available
at as_of, respect observation periods and target units, and consider all asset /
horizon cells jointly. A log_return target is a cumulative log return, not a price,
percentage or future daily return. Return only {"forecast":[{"cell_id":"c0",
"value":number}, ...]} with exactly one entry for every supplied target cell.
"""


def without_forecast(context):
    # Allowlist rather than deleting a few known quantiles: future additions of
    # rule-derived metadata must not leak into this control.
    result = {k: copy.deepcopy(context[k]) for k in
              ('as_of', 'target_type', 'value_unit', 'horizon_unit', 'histories')}
    result['cells'] = [{k: copy.deepcopy(c[k]) for k in
                       ('cell_id', 'asset', 'horizon', 'observation_period')}
                      for c in context['cells']]
    return result


def raw_history_fallback(context):
    if context['target_type'] == 'log_return':
        return np.zeros(len(context['cells']))
    if context['target_type'] != 'level':
        raise ValueError('No declared raw-history fallback for this target type')
    return np.array([context['histories'][c['asset']]['last_value'] for c in context['cells']])


def direct_paths(context, docs, call, *, news=True):
    ctx = without_forecast(context)
    text = DIRECT+'\n'+payload(ctx, docs if news else [])
    paths, errors = [], []
    for i in range(25):
        parsed = call(f'direct_{i:02d}', PATH_SYSTEM, text, 1., i)
        try:
            row = parse_trajectory(parsed, ctx)
        except ValueError as exc:
            errors.append({'path': i, 'error': str(exc)})
            row = raw_history_fallback(ctx)
        paths.append(row)
    return empirical_draws(paths), {'calls': 25, 'errors': errors, 'trajectories': np.asarray(paths).tolist(),
        'rule_based_forecast_in_prompt': False,
        'invalid_reply_policy': 'raw-history last level or zero cumulative return; never rule-based forecast'}


def single_median(x, context, docs, call):
    raw = call('direct_median', cordp.SYSTEM, cordp.CORRECT+'\n'+payload(context, docs), 0., 0)
    audit = {'calls': 1, 'errors': [], 'response': raw}
    try:
        median = cordp.correction_values(raw, context, docs)
    except ValueError as exc:
        audit['errors'].append(str(exc))
        return x.copy(), audit
    return cordp.translate(x, median), audit


def ungated_transport(x, context, docs, call, *, reviewed=False):
    ctx = transport_context(x, context)
    instruction = PROPOSE.replace("supported by the gate's evidence", "supported by the supplied dated documents")
    proposal = call('propose', SYSTEM, instruction+'\n'+payload(ctx, docs), 0., 0)
    audit = {'calls': 1, 'errors': [], 'proposal': proposal, 'gate': None}
    final = proposal
    if reviewed:
        review = call('review', SYSTEM, REVIEW+'\n'+payload(ctx, docs, proposal=proposal), 0., 0)
        final = call('final', SYSTEM, instruction+'\nResolve the review issues before returning the final maps.\n'+
                     payload(ctx, docs, proposal=proposal, review=review), 0., 0)
        audit.update(calls=3, review=review, final=final)
    try:
        target = parse_map(final, ctx, docs)
        source = np.asarray([c['source_quantiles'] for c in ctx['cells']])
        result = transport_samples(x, source, target)
    except ValueError as exc:
        audit['errors'].append(str(exc))
        return x.copy(), audit
    audit.update(source=source.tolist(), target=target.tolist(), applied=bool(np.any(result != x)))
    return result, audit
