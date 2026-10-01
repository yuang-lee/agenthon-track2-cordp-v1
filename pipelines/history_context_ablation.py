"""## Executive summary (read this first)

Vary only the numerical history length in an existing forecast context.
Keep news, forecast targets, dates and baseline parameters unchanged. Read only
observations at or before the task cutoff; do not read forecast outcomes.
"""
from copy import deepcopy

import pandas as pd

from engine import _full_series


def with_history_length(context, unit, observations):
    if isinstance(observations, bool) or not isinstance(observations, int) or observations < 1:
        raise ValueError('History length must be a positive integer')
    result = deepcopy(context)
    for asset, history in result['histories'].items():
        series, _ = _full_series(unit, asset, pd.Timestamp(context['as_of']))
        if str(series.index[-1].date()) != history['last_observation']:
            raise ValueError('Last observed date differs from the frozen context')
        if float(series.iloc[-1]) != history['last_value']:
            raise ValueError('Last observed value differs from the frozen context')
        if any(date > pd.Timestamp(context['as_of']) for date in series.index):
            raise ValueError('Post-cutoff observation')
        history['recent_observations'] = [
            {'date': str(date.date()), 'value': float(value)}
            for date, value in series.iloc[-observations:].items()]
    return result
