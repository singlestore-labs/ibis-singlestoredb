from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class ScalarVectorMul(Value):
    """32-bit float dot product."""

    arg = binary
    n = rlz.strict_numeric

    output_dtype = dt.binary
    output_shape = rlz.shape_like('arg')


def scalar_vector_mul(arg: ir.BinaryValue, n: ir.NumericValue) -> ir.BinaryValue:
    """
    Multiply each element in the 32-bit float vector blob by the specified value.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int or float
        Multiplier value

    Returns
    -------
    Blob column

    """
    return ScalarVectorMul(arg, n).to_expr()


ir.BinaryValue.scalar_vector_mul = scalar_vector_mul


class ScalarVectorMulI8(ScalarVectorMul):
    """8-bit integer dot product."""


def scalar_vector_mul_i8(arg: ir.BinaryValue, n: ir.NumericValue) -> ir.BinaryValue:
    """
    Multiply each element in the 8-bit int vector blob by the specified value.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int or float
        Multiplier value

    Returns
    -------
    Blob column

    """
    return ScalarVectorMulI8(arg, n).to_expr()


ir.BinaryValue.scalar_vector_mul_i8 = scalar_vector_mul_i8


class ScalarVectorMulI16(ScalarVectorMul):
    """16-bit integer dot product."""


def scalar_vector_mul_i16(arg: ir.BinaryValue, n: ir.NumericValue) -> ir.BinaryValue:
    """
    Multiply each element in the 16-bit int vector blob by the specified value.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int or float
        Multiplier value

    Returns
    -------
    Blob column

    """
    return ScalarVectorMulI16(arg, n).to_expr()


ir.BinaryValue.scalar_vector_mul_i16 = scalar_vector_mul_i16


class ScalarVectorMulI32(ScalarVectorMul):
    """32-bit integer dot product."""


def scalar_vector_mul_i32(arg: ir.BinaryValue, n: ir.NumericValue) -> ir.BinaryValue:
    """
    Multiply each element in the 32-bit int vector blob by the specified value.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int or float
        Multiplier value

    Returns
    -------
    Blob column

    """
    return ScalarVectorMulI32(arg, n).to_expr()


ir.BinaryValue.scalar_vector_mul_i32 = scalar_vector_mul_i32


class ScalarVectorMulI64(ScalarVectorMul):
    """64-bit integer dot product."""


def scalar_vector_mul_i64(arg: ir.BinaryValue, n: ir.NumericValue) -> ir.BinaryValue:
    """
    Multiply each element in the 64-bit int vector blob by the specified value.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int or float
        Multiplier value

    Returns
    -------
    Blob column

    """
    return ScalarVectorMulI64(arg, n).to_expr()


ir.BinaryValue.scalar_vector_mul_i64 = scalar_vector_mul_i64


class ScalarVectorMulF32(ScalarVectorMul):
    """32-bit float dot product."""


def scalar_vector_mul_f32(arg: ir.BinaryValue, n: ir.NumericValue) -> ir.BinaryValue:
    """
    Multiply each element in the 32-bit float vector blob by the specified value.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int or float
        Multiplier value

    Returns
    -------
    Blob column

    """
    return ScalarVectorMulF32(arg, n).to_expr()


ir.BinaryValue.scalar_vector_mul_f32 = scalar_vector_mul_f32


class ScalarVectorMulF64(ScalarVectorMul):
    """64-bit float dot product."""


def scalar_vector_mul_f64(arg: ir.BinaryValue, n: ir.NumericValue) -> ir.BinaryValue:
    """
    Multiply each element in the 64-bit float vector blob by the specified value.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int or float
        Multiplier value

    Returns
    -------
    Blob column

    """
    return ScalarVectorMulF64(arg, n).to_expr()


ir.BinaryValue.scalar_vector_mul_f64 = scalar_vector_mul_f64
