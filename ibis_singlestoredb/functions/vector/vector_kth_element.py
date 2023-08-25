from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorKthElement(Value):
    """32-bit float kth element."""

    arg = binary
    n = rlz.integer

    output_dtype = dt.float32
    output_shape = rlz.shape_like('arg')


def vector_kth_element(arg: ir.BinaryValue, n: ir.IntegerValue) -> ir.FloatingValue:
    """
    Return the 32-bit float kth element in a zero-indexed vector expression.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int
        Element index

    Returns
    -------
    32-bit float column

    """
    return VectorKthElement(arg, n).to_expr()


ir.BinaryValue.vector_kth_element = vector_kth_element


class VectorKthElementI8(VectorKthElement):
    """8-bit integer kth element."""

    output_dtype = dt.int8


def vector_kth_element_i8(arg: ir.BinaryValue, n: ir.IntegerValue) -> ir.IntegerValue:
    """
    Return the 8-bit int kth element in a zero-indexed vector expression.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int
        Element index

    Returns
    -------
    8-bit int column

    """
    return VectorKthElementI8(arg, n).to_expr()


ir.BinaryValue.vector_kth_element_i8 = vector_kth_element_i8


class VectorKthElementI16(VectorKthElement):
    """16-bit integer kth element."""

    output_dtype = dt.int16


def vector_kth_element_i16(arg: ir.BinaryValue, n: ir.IntegerValue) -> ir.IntegerValue:
    """
    Return the 16-bit int kth element in a zero-indexed vector expression.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int
        Element index

    Returns
    -------
    16-bit int column

    """
    return VectorKthElementI16(arg, n).to_expr()


ir.BinaryValue.vector_kth_element_i16 = vector_kth_element_i16


class VectorKthElementI32(VectorKthElement):
    """32-bit integer kth element."""

    output_dtype = dt.int32


def vector_kth_element_i32(arg: ir.BinaryValue, n: ir.IntegerValue) -> ir.IntegerValue:
    """
    Return the 32-bit int kth element in a zero-indexed vector expression.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int
        Element index

    Returns
    -------
    32-bit int column

    """
    return VectorKthElementI32(arg, n).to_expr()


ir.BinaryValue.vector_kth_element_i32 = vector_kth_element_i32


class VectorKthElementI64(VectorKthElement):
    """64-bit integer kth element."""

    output_dtype = dt.int64


def vector_kth_element_i64(arg: ir.BinaryValue, n: ir.IntegerValue) -> ir.IntegerValue:
    """
    Return the 64-bit int kth element in a zero-indexed vector expression.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int
        Element index

    Returns
    -------
    64-bit int column

    """
    return VectorKthElementI64(arg, n).to_expr()


ir.BinaryValue.vector_kth_element_i64 = vector_kth_element_i64


class VectorKthElementF32(VectorKthElement):
    """32-bit float kth element."""

    output_dtype = dt.float32


def vector_kth_element_f32(arg: ir.BinaryValue, n: ir.IntegerValue) -> ir.FloatingValue:
    """
    Return the 32-bit float kth element in a zero-indexed vector expression.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int
        Element index

    Returns
    -------
    32-bit float column

    """
    return VectorKthElementF32(arg, n).to_expr()


ir.BinaryValue.vector_kth_element_f32 = vector_kth_element_f32


class VectorKthElementF64(VectorKthElement):
    """64-bit float kth element."""

    output_dtype = dt.float64


def vector_kth_element_f64(arg: ir.BinaryValue, n: ir.IntegerValue) -> ir.FloatingValue:
    """
    Return the 64-bit float kth element in a zero-indexed vector expression.

    Parameters
    ----------
    arg : blob
        Vector expression
    n : int
        Element index

    Returns
    -------
    64-bit float column

    """
    return VectorKthElementF64(arg, n).to_expr()


ir.BinaryValue.vector_kth_element_f64 = vector_kth_element_f64
