#!/usr/bin/env python3
from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations import BinaryOp
from ibis.expr.types import Expr
from ibis.expr.types import IntegerValue
from ibis.expr.types import StringValue
from public import public


@public
class StrCmp(BinaryOp):
    left = rlz.string
    right = rlz.string

    output_dtype = dt.int8
    output_shape = rlz.shape_like('left')


def strcmp(left_string_value: rlz.string, right_string_value: rlz.string) -> Expr:
    return StrCmp(left_string_value, right_string_value).to_expr()


StringValue.strcmp = strcmp


@public
class PowerOf(BinaryOp):
    left = rlz.integer
    right = rlz.integer

    output_dtype = dt.int64
    output_shape = rlz.shape_like('left')


def power_of(left_int_value: rlz.integer, right_int_value: rlz.integer) -> Expr:
    return PowerOf(left_int_value, right_int_value).to_expr()


IntegerValue.power_of = power_of


@public
class EvalXpath(BinaryOp):
    left = rlz.string
    right = rlz.string

    output_dtype = dt.string
    output_shape = rlz.shape_like('left')


def eval_xpath(left_value: rlz.string, right_value: rlz.string) -> Expr:
    return eval_xpath(left_value, right_value).to_expr()


StringValue.eval_xpath = eval_xpath
