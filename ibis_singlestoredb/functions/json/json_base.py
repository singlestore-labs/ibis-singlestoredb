from __future__ import annotations

from typing import Any
from typing import Tuple

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir


class JSONGetPath(ops.Value):
    """JSON get path."""
    arg = rlz.one_of([rlz.json, rlz.string])
    index = rlz.tuple_of(rlz.any, min_length=1)

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def __getitem__(
    self: ir.JSONValue | ir.StringValue,
    key: Tuple[Any, ...],
) -> ir.JSONValue | ir.StringValue:
    if isinstance(key, tuple):
        return JSONGetPath(self, key).to_expr()
    return ops.JSONGetItem(self, key).to_expr()


ir.JSONValue.__getitem__ = __getitem__
