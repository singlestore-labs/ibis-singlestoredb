from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorElementsSum(Value):
    """32-bit float JSON array unpacker."""

    arg = binary

    output_dtype = dt.float64
    output_shape = rlz.shape_like('arg')


def vector_elements_sum(arg: ir.BinaryValue) -> ir.FloatingValue:
    """
    Convert an encoded blob representing a 32-bit float vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return VectorElementsSum(arg).to_expr()


ir.BinaryValue.vector_elements_sum = vector_elements_sum


class VectorElementsSumI8(VectorElementsSum):
    """8-bit integer JSON array unpacker."""

    output_dtype = dt.int64


def vector_elements_sum_i8(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Convert an encoded blob representing an 8-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return VectorElementsSumI8(arg).to_expr()


ir.BinaryValue.vector_elements_sum_i8 = vector_elements_sum_i8


class VectorElementsSumI16(VectorElementsSum):
    """16-bit integer JSON array unpacker."""

    output_dtype = dt.int64


def vector_elements_sum_i16(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Convert an encoded blob representing a 16-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return VectorElementsSumI16(arg).to_expr()


ir.BinaryValue.vector_elements_sum_i16 = vector_elements_sum_i16


class VectorElementsSumI32(VectorElementsSum):
    """32-bit integer JSON array unpacker."""

    output_dtype = dt.int64


def vector_elements_sum_i32(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Convert an encoded blob representing a 32-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return VectorElementsSumI32(arg).to_expr()


ir.BinaryValue.vector_elements_sum_i32 = vector_elements_sum_i32


class VectorElementsSumI64(VectorElementsSum):
    """64-bit integer JSON array unpacker."""

    output_dtype = dt.int64


def vector_elements_sum_i64(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Convert an encoded blob representing a 64-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return VectorElementsSumI64(arg).to_expr()


ir.BinaryValue.vector_elements_sum_i64 = vector_elements_sum_i64


class VectorElementsSumF32(VectorElementsSum):
    """32-bit float JSON array unpacker."""

    output_dtype = dt.double


def vector_elements_sum_f32(arg: ir.BinaryValue) -> ir.FloatingValue:
    """
    Convert an encoded blob representing a 32-bit float vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return VectorElementsSumF32(arg).to_expr()


ir.BinaryValue.vector_elements_sum_f32 = vector_elements_sum_f32


class VectorElementsSumF64(VectorElementsSum):
    """64-bit float JSON array unpacker."""

    output_dtype = dt.double


def vector_elements_sum_f64(arg: ir.BinaryValue) -> ir.FloatingValue:
    """
    Convert an encoded blob representing a 64-bit float vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return VectorElementsSumF64(arg).to_expr()


ir.BinaryValue.vector_elements_sum_f64 = vector_elements_sum_f64
