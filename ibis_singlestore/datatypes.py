from __future__ import annotations

from functools import partial

import ibis.expr.datatypes as dt

# binary character set
# used to distinguish blob binary vs blob text
MY_CHARSET_BIN = 63


def _type_from_cursor_info(descr) -> dt.DataType:
    """Construct an ibis type from SingleStore field descr and field result metadata.

    This method is complex because the SingleStore protocol is complex.

    Types are not encoded in a self contained way, meaning you need
    multiple pieces of information coming from the result set metadata to
    determine the most precise type for a field. Even then, the decoding is
    not high fidelity in some cases: UUIDs for example are decoded as
    strings, because the protocol does not appear to preserve the logical
    type, only the physical type.
    """
    BIT = 16
    BLOB = 252
    LONG_BLOB = 251
    MEDIUM_BLOB = 250
    STRING = 254
    TINY_BLOB = 249
    VAR_STRING = 253
    VARCHAR = 15
    GEOMETRY = 255
    
    TEXT_TYPES = {
        BIT,
        BLOB,
        LONG_BLOB,
        MEDIUM_BLOB,
        STRING,
        TINY_BLOB,
        VAR_STRING,
        VARCHAR,
        GEOMETRY,
    }

    _, type_code, _, _, field_length, scale, _, flags, encoding = descr
    flags = _FieldFlags(flags)
    typename = _type_codes.get(type_code)
    if typename is None:
        raise NotImplementedError(
            f"SingleStore type code {type_code:d} is not supported"
        )

    typ = _type_mapping[typename]

    if typename in ("DECIMAL", "NEWDECIMAL"):
        precision = _decimal_length_to_precision(
            length=field_length,
            scale=scale,
            is_unsigned=flags.is_unsigned,
        )
        typ = partial(typ, precision=precision, scale=scale)
    elif typename == "BIT":
        if field_length is None:
            typ = dt.int64
        else:
            if field_length <= 8:
                typ = dt.int8
            elif field_length <= 16:
                typ = dt.int16
            elif field_length <= 32:
                typ = dt.int32
            elif field_length <= 64:
                typ = dt.int64
            else:
                assert False, "invalid field length for BIT type"
    else:
        if flags.is_set:
            # sets are limited to strings
            typ = dt.Set(dt.string)
        elif flags.is_unsigned and flags.is_num:
            typ = getattr(dt, f"U{typ.__name__}")
        elif type_code in TEXT_TYPES:
            # binary text
            if encoding == MY_CHARSET_BIN:
                typ = dt.Binary
            else:
                typ = dt.String

    # projection columns are always nullable
    return typ(nullable=True)


# ported from my_decimal.h:my_decimal_length_to_precision in mariadb
def _decimal_length_to_precision(
    *,
    length: int,
    scale: int,
    is_unsigned: bool,
) -> int:
    if length is None or scale is None:
        return None
    return length - (scale > 0) - (not (is_unsigned or not length))


_type_codes = {
    0: "DECIMAL",
    1: "TINY",
    2: "SHORT",
    3: "LONG",
    4: "FLOAT",
    5: "DOUBLE",
    6: "NULL",
    7: "TIMESTAMP",
    8: "LONGLONG",
    9: "INT24",
    10: "DATE",
    11: "TIME",
    12: "DATETIME",
    13: "YEAR",
    15: "VARCHAR",
    16: "BIT",
    245: "JSON",
    246: "NEWDECIMAL",
    247: "ENUM",
    248: "SET",
    249: "TINY_BLOB",
    250: "MEDIUM_BLOB",
    251: "LONG_BLOB",
    252: "BLOB",
    253: "VAR_STRING",
    254: "STRING",
    255: "GEOMETRY",
}


_type_mapping = {
    "DECIMAL": dt.Decimal,
    "TINY": dt.Int8,
    "SHORT": dt.Int16,
    "LONG": dt.Int32,
    "FLOAT": dt.Float32,
    "DOUBLE": dt.Float64,
    "NULL": dt.Null,
    "TIMESTAMP": lambda nullable: dt.Timestamp(
        timezone="UTC",
        nullable=nullable,
    ),
    "LONGLONG": dt.Int64,
    "INT24": dt.Int32,
    "DATE": dt.Date,
    "TIME": dt.Time,
    "DATETIME": dt.Timestamp,
    "YEAR": dt.Int8,
    "VARCHAR": dt.String,
    "BIT": dt.Int8,
    "JSON": dt.JSON,
    "NEWDECIMAL": dt.Decimal,
    "ENUM": dt.String,
    "SET": lambda nullable: dt.Set(dt.string, nullable=nullable),
    "TINY_BLOB": dt.Binary,
    "MEDIUM_BLOB": dt.Binary,
    "LONG_BLOB": dt.Binary,
    "BLOB": dt.Binary,
    "VAR_STRING": dt.String,
    "STRING": dt.String,
    "GEOMETRY": dt.Geometry,
}


class _FieldFlags:
    """Flags used to disambiguate field types.

    Gaps in the flag numbers are because we do not map in flags that are of no
    use in determining the field's type, such as whether the field is a primary
    key or not.
    """

    UNSIGNED = 1 << 5
    SET = 1 << 11
    NUM = 1 << 15

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    @property
    def is_unsigned(self) -> bool:
        return (self.UNSIGNED & self.value) != 0

    @property
    def is_set(self) -> bool:
        return (self.SET & self.value) != 0

    @property
    def is_num(self) -> bool:
        return (self.NUM & self.value) != 0
