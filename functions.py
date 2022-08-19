from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations import BinaryOp

@public
class StrCmp(BinaryOp):
    left = rlz.string
    right = rlz.string

    output_dtype = dt.int8
    output_shape = rlz.shape_like('left')

from ibis.expr.types import StringValue

def strcmp(left_string_value, right_string_value):
    return StrCmp(left_string_value, right_string_value).to_expr()


StringValue.strcmp = strcmp


@public
class PowerOf(BinaryOp):
    left = rlz.integer
    right = rlz.integer

    output_dtype = dt.int64
    output_shape = rlz.shape_like('left')

from ibis.expr.types import IntegerValue

def power_of(left_int_value, right_int_value):
    return PowerOf(left_int_value, right_int_value).to_expr()

IntegerValue.power_of = power_of

@public
class EvalXpath(BinaryOp):
    left = rlz.string
    right = rlz.string

    output_dtype = dt.string
    output_shape = rlz.shape_like('left')

from ibis.expr.types import StringValue

def eval_path(left_value, right_value):
    return eval_path(left_value, right_value).to_expr()

StringValue.eval_path = eval_path