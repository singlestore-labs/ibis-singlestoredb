from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

from .json_array_contains import json_array_contains


class JSONKeys(Value):
    """JSON exclude using a mask."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]))

    output_dtype = dt.json
    output_shape = rlz.shape_like('arg')


def json_keys(
    arg: ir.JSONValue | ir.StringValue, *key_path: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Return the top-level keys of a JSON object in the form of a JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Path to a JSON object

    Returns
    -------
    JSON or String column

    """
    return JSONKeys(arg, key_path).to_expr()


def json_has_key(
    self: ir.JSONValue | ir.StringValue,
    *key_path: str | int | ir.StringValue | ir.IntegerValue,
) -> ir.BooleanValue:
    assert key_path, '`key_path` must contain at least one element'
    return json_array_contains(self.keys(*key_path[:-1]), key_path[-1])


ir.JSONValue.has_key = json_has_key
ir.JSONValue.keys = json_keys
ir.JSONValue.json_keys = json_keys
ir.StringValue.json_keys = json_keys
ir.StringValue.json_has_key = json_has_key
