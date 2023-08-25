from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONLength(Value):
    """JSON object / array length."""

    arg = rlz.one_of([rlz.json, rlz.string])

    output_dtype = dt.int
    output_shape = rlz.shape_like('arg')


def json_length(arg: ir.JSONValue | ir.StringValue) -> ir.IntegerValue:
    """
    Get the length of the JSON object / array.

    Parameters
    ----------
    arg : JSON or string
        JSON array

    Returns
    -------
    Integer column

    """
    return JSONLength(arg).to_expr()


ir.JSONValue.json_length = json_length
ir.StringValue.json_length = json_length
