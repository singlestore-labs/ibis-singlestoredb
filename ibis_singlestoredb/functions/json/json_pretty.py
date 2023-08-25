from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONPretty(Value):
    """Pretty-print JSON."""

    arg = rlz.one_of([rlz.json, rlz.string])

    output_dtype = dt.string
    output_shape = rlz.shape_like('arg')


def json_pretty(arg: ir.JSONValue | ir.StringValue) -> ir.StringValue:
    """
    Pretty-print a JSON object or array.

    Parameters
    ----------
    arg : JSON or string
        JSON array

    Returns
    -------
    String column

    """
    return JSONPretty(arg).to_expr()


ir.JSONValue.json_pretty = json_pretty
ir.StringValue.json_pretty = json_pretty
