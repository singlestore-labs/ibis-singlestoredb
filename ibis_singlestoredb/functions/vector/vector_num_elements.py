from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorNumElements(Value):
    """32-bit float kth element."""

    arg = binary

    output_dtype = dt.int
    output_shape = rlz.shape_like('arg')


def vector_num_elements(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Return the number of elements in a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Integer column

    """
    return VectorNumElements(arg).to_expr()


ir.BinaryValue.vector_num_elements = vector_num_elements


class VectorNumElementsI8(VectorNumElements):
    """8-bit integer kth element."""


def vector_num_elements_i8(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Return the number of elements in an 8-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Integer column

    """
    return VectorNumElementsI8(arg).to_expr()


ir.BinaryValue.vector_num_elements_i8 = vector_num_elements_i8


class VectorNumElementsI16(VectorNumElements):
    """16-bit integer kth element."""


def vector_num_elements_i16(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Return the number of elements in a 16-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Integer column

    """
    return VectorNumElementsI16(arg).to_expr()


ir.BinaryValue.vector_num_elements_i16 = vector_num_elements_i16


class VectorNumElementsI32(VectorNumElements):
    """32-bit integer kth element."""


def vector_num_elements_i32(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Return the number of elements in a 32-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Integer column

    """
    return VectorNumElementsI32(arg).to_expr()


ir.BinaryValue.vector_num_elements_i32 = vector_num_elements_i32


class VectorNumElementsI64(VectorNumElements):
    """64-bit integer kth element."""


def vector_num_elements_i64(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Return the number of elements in a 64-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Integer column

    """
    return VectorNumElementsI64(arg).to_expr()


ir.BinaryValue.vector_num_elements_i64 = vector_num_elements_i64


class VectorNumElementsF32(VectorNumElements):
    """32-bit float kth element."""


def vector_num_elements_f32(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Return the number of elements in a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Integer column

    """
    return VectorNumElementsF32(arg).to_expr()


ir.BinaryValue.vector_num_elements_f32 = vector_num_elements_f32


class VectorNumElementsF64(VectorNumElements):
    """64-bit float kth element."""


def vector_num_elements_f64(arg: ir.BinaryValue) -> ir.IntegerValue:
    """
    Return the number of elements in a 64-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression

    Returns
    -------
    Integer column

    """
    return VectorNumElementsF64(arg).to_expr()


ir.BinaryValue.vector_num_elements_f64 = vector_num_elements_f64
