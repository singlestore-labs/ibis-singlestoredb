from __future__ import annotations

import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONDeleteKey(Value):
    """JSON delete key."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_delete_key(
    arg: ir.JSONValue | ir.StringValue,
    *key_path: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Delete a key from a JSON map or array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Keys or zero-indexed array positions.

    Returns
    -------
    JSON or string column

    """
    return JSONDeleteKey(arg, key_path).to_expr()


ir.JSONValue.json_delete_key = json_delete_key
ir.StringValue.json_delete_key = json_delete_key
