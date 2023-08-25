from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class Hex(Value):
    """Convert bytes to hex values."""

    arg = binary

    output_dtype = dt.string
    output_shape = rlz.shape_like('arg')


def hex(arg: ir.BinaryValue) -> ir.StringValue:
    """
    Convert a collection of bytes into hex values.

    Parameters
    ----------
    arg : binary
        Blob to convert

    Returns
    -------
    String column

    """
    return Hex(arg).to_expr()


ir.BinaryValue.hex = hex


class Unhex(Value):
    """Convert hex string to bytes."""

    arg = rlz.string

    output_dtype = dt.binary
    output_shape = rlz.shape_like('arg')


def unhex(arg: ir.StringValue) -> ir.BinaryValue:
    """
    Convert a string of hex values to bytes.

    Parameters
    ----------
    arg : string
        Hex string to convert

    Returns
    -------
    Binary column

    """
    return Unhex(arg).to_expr()


ir.StringValue.unhex = unhex
