"""## Executive summary (read this first)

Constrain forecast serialization to one numeric value per declared target cell.
The schema fixes only identifiers and structure, never numeric forecast values,
ranges or outcomes. It is a local serving format control to validate separately.
"""


def forecast_schema(context):
    ids = [c['cell_id'] for c in context['cells']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Require a nonempty unique target grid')
    entries = [{'type':'object', 'properties':{'cell_id':{'const':cell}, 'value':{'type':'number'}},
                'required':['cell_id', 'value'], 'additionalProperties':False} for cell in ids]
    return {'type':'object', 'properties':{'forecast':{'type':'array', 'prefixItems':entries,
        'items':False, 'minItems':len(ids), 'maxItems':len(ids)}},
        'required':['forecast'], 'additionalProperties':False}
