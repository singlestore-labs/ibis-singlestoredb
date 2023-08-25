from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Reduction

binary = rlz.value(dt.binary)


class VectorSum(Reduction):
    """32-bit float columnar vector sum."""

    arg = binary

    output_dtype = dt.binary


def vector_sum(arg: ir.BinaryValue) -> ir.BinaryValue:
    """
    Return a columnar sum of 32-bit float vectors.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSum(arg).to_expr()


ir.BinaryValue.vector_sum = vector_sum


class VectorSumI8(VectorSum):
    """8-bit integer columnar vector sum."""


def vector_sum_i8(arg: ir.BinaryValue) -> ir.BinaryValue:
    """
    Return a columnar sum of 8-bit int vectors.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSumI8(arg).to_expr()


ir.BinaryValue.vector_sum_i8 = vector_sum_i8


class VectorSumI16(VectorSum):
    """16-bit integer columnar vector sum."""


def vector_sum_i16(arg: ir.BinaryValue) -> ir.BinaryValue:
    """
    Return a columnar sum of 16-bit int vectors.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSum(arg).to_expr()


ir.BinaryValue.vector_sum_i16 = vector_sum_i16


class VectorSumI32(VectorSum):
    """32-bit integer columnar vector sum."""


def vector_sum_i32(arg: ir.BinaryValue) -> ir.BinaryValue:
    """
    Return a columnar sum of 32-bit int vectors.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSumI32(arg).to_expr()


ir.BinaryValue.vector_sum_i32 = vector_sum_i32


class VectorSumI64(VectorSum):
    """64-bit integer columnar vector sum."""


def vector_sum_i64(arg: ir.BinaryValue) -> ir.BinaryValue:
    """
    Return a columnar sum of 64-bit int vectors.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSumI64(arg).to_expr()


ir.BinaryValue.vector_sum_i64 = vector_sum_i64


class VectorSumF32(VectorSum):
    """32-bit float columnar vector sum."""


def vector_sum_f32(arg: ir.BinaryValue) -> ir.BinaryValue:
    """
    Return a columnar sum of 32-bit float vectors.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSumF32(arg).to_expr()


ir.BinaryValue.vector_sum_f32 = vector_sum_f32


class VectorSumF64(VectorSum):
    """64-bit float columnar vector sum."""


def vector_sum_f64(arg: ir.BinaryValue) -> ir.BinaryValue:
    """
    Return a columnar sum of 64-bit float vectors.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSumF64(arg).to_expr()


ir.BinaryValue.vector_sum_f64 = vector_sum_f64
