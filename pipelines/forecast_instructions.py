"""## Executive summary (read this first)

Vary only the instruction defining requested numerical targets. Keep histories,
news, schema, system prompt, sampling and the25-call protocol unchanged. Distinguish
future levels from changes, and cumulative log returns from one-day simple returns.
"""
from agentic_ablations import DIRECT, direct_paths


RETURN_CLAUSE = ('A log_return target is a cumulative log return, not a price,\n'
                 'percentage or future daily return.')


def direct_instruction(context, variant='original'):
    if variant == 'original':
        return DIRECT
    if variant == 'scenario':
        return (DIRECT + '\nGenerate one plausible joint future realization sampled '
            'from the predictive distribution, rather than a central estimate. '
            'Represent uncertainty in future outcomes, not uncertainty about which '
            'central estimate to choose. Each call is one scenario. Keep values '
            'coherent across assets and forecast horizons.\n')
    if variant != 'target_contract':
        raise ValueError('Unknown forecasting instruction variant')
    target = context['target_type']
    if target == 'level':
        clause = (
            'These targets are FUTURE LEVELS. Output the actual future observed value '
            'in the same numerical scale and units as that asset\'s history, NOT its '
            'change, return, percentage change, standardized value or direction. '
            'Different assets can have very different numerical scales. Currency '
            'exchange-rate levels must be strictly positive; interest rates may be negative.')
    elif target == 'log_return':
        clause = (
            'These targets are cumulative log returns over the FUTURE horizon. '
            'The supplied factor history contains one-period simple returns as decimals. '
            'The target sums log(1+r) over future periods; do not output a future '
            'single-period return, price level or percentage-number representation.')
    else:
        raise ValueError('Unknown forecast target type')
    if RETURN_CLAUSE not in DIRECT:
        raise ValueError('Original instruction changed; re-audit this intervention')
    return DIRECT.replace(RETURN_CLAUSE, clause)


def instructed_paths(context, docs, call, *, news=True, variant='original'):
    replacement = direct_instruction(context, variant)
    def wrapped(role, system, text, temperature, seed):
        if not text.startswith(DIRECT+'\n'):
            raise ValueError('Unexpected direct-forecast prompt layout')
        return call(role, system, replacement + text[len(DIRECT):], temperature, seed)
    draws, audit = direct_paths(context, docs, wrapped, news=news)
    return draws, {**audit, 'instruction_variant': variant}
