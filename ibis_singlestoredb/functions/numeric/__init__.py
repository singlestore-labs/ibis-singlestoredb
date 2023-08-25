from __future__ import annotations

from typing import Optional

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class BitCount(Value):
    """Number of bits required to construct a number."""

    arg = rlz.strict_numeric

    output_dtype = dt.int
    output_shape = rlz.shape_like('arg')


def bit_count(arg: ir.NumericValue) -> ir.IntegerValue:
    """
    Return the number of ones in the binary representation of a given number.

    Parameters
    ----------
    arg : int or float
        The number to compute the ones in

    Returns
    -------
    IntegerValue

    """
    return BitCount(arg).to_expr()


ir.NumericValue.bit_count = bit_count


class Conv(Value):
    """Numeric base converter."""

    arg = rlz.one_of([rlz.integer, rlz.string])
    from_base = rlz.integer
    to_base = rlz.integer

    output_dtype = dt.string
    output_shape = rlz.shape_like('arg')


def conv(
    arg: ir.NumericValue, from_base: ir.IntegerValue, to_base: ir.IntegerValue,
) -> ir.IntegerValue:
    """
    Convert number between different bases.

    Parameters
    ----------
    arg : int
        The number to convert
    from_base : int
        Base of input value
    to_base : int
        Base of output value

    Returns
    -------
    IntegerValue

    """
    return Conv(arg, from_base, to_base).to_expr()


ir.NumericValue.conv = conv


class Sigmoid(Value):
    """Sigmoid function."""

    arg = rlz.strict_numeric

    output_dtype = dt.double
    output_shape = rlz.shape_like('arg')


def sigmoid(arg: ir.NumericValue) -> ir.FloatingValue:
    """
    Compute the sigmoid function of `arg`.

    Parameters
    ----------
    arg : int or float
        The number to use

    Returns
    -------
    FloatingValue

    """
    return Sigmoid(arg).to_expr()


ir.NumericValue.sigmoid = sigmoid


class ToNumber(Value):
    """Decimal parser."""

    arg = rlz.string
    format_string = rlz.optional(rlz.string)

    output_dtype = dt.double
    output_shape = rlz.shape_like('arg')


def to_number(
    arg: ir.StringValue, format_string: Optional[ir.StringValue] = None,
) -> ir.DecimalValue:
    """
    Convert a string to a decimal value.

    Parameters
    ----------
    arg : string
        The string value to convert
    format_string : string, optional
        The format of the input value

    Returns
    -------
    DecimalValue

    """
    return ToNumber(arg, format_string).to_expr()


ir.StringValue.to_number = to_number


class Trunc(Value):
    """Truncate decimals."""

    arg = rlz.strict_numeric
    decimals = rlz.optional(rlz.integer)

    output_dtype = dt.double
    output_shape = rlz.shape_like('arg')


def trunc(
    arg: ir.NumericValue, decimals: Optional[ir.IntegerValue] = None,
) -> ir.NumericValue:
    """
    Truncate the number of decimals to `decimals`.

    Parameters
    ----------
    arg : float
        The float value to truncate
    decimals : int, optional
        The number of requested decimals

    Returns
    -------
    NumericValue

    """
    return Trunc(arg, decimals).to_expr()


ir.NumericValue.trunc = trunc


class Truncate(Value):
    """Truncate decimals."""

    arg = rlz.strict_numeric
    decimals = rlz.integer

    output_dtype = dt.double
    output_shape = rlz.shape_like('arg')


def truncate(arg: ir.NumericValue, decimals: ir.IntegerValue) -> ir.NumericValue:
    """
    Truncate the number of decimals to `decimals`.

    Parameters
    ----------
    arg : float
        The float value to truncate
    decimals : int, optional
        The number of requested decimals

    Returns
    -------
    NumericValue

    """
    return Truncate(arg, decimals).to_expr()


ir.NumericValue.truncate = truncate
