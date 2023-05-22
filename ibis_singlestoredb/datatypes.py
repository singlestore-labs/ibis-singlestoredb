from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Tuple

import ibis.expr.datatypes as dt
import sqlalchemy as sa
import sqlalchemy_singlestoredb as singlestoredb
from ibis.backends.base.sql.alchemy import to_sqla_type
from sqlalchemy_singlestoredb.base import SingleStoreDBDialect

# binary character set
# used to distinguish blob binary vs blob text
MY_CHARSET_BIN = 63


def _type_from_cursor_info(descr: Tuple[Any, ...], field: Any) -> dt.DataType:
    """Construct an ibis type from SingleStoreDB field descr and field result metadata.

    This method is complex because the SingleStoreDB protocol is complex.

    Types are not encoded in a self contained way, meaning you need
    multiple pieces of information coming from the result set metadata to
    determine the most precise type for a field. Even then, the decoding is
    not high fidelity in some cases: UUIDs for example are decoded as
    strings, because the protocol does not appear to preserve the logical
    type, only the physical type.
    """
    _, type_code, _, _, field_length, scale, _, _, _ = descr
    flags = _FieldFlags(field.flags)
    typename = _type_codes.get(type_code)
    if typename is None:
        raise NotImplementedError(
            f'SingleStoreDB type code {type_code:d} is not supported',
        )

    if typename in ('DECIMAL', 'NEWDECIMAL'):
        precision = _decimal_length_to_precision(
            length=field_length,
            scale=scale,
            is_unsigned=flags.is_unsigned,
        )
        typ = partial(_type_mapping[typename], precision=precision, scale=scale)

    elif typename == 'BIT':
        typ = dt.int64

    elif typename == 'YEAR':
        typ = dt.uint8

    elif flags.is_set:
        # sets are limited to strings
        typ = dt.Array(dt.string)

    elif flags.is_unsigned and type_code in _num_types:
        typ = getattr(dt, f'U{_type_mapping[typename].__name__}')

    elif type_code in _char_types:
        # binary text
        if field.charsetnr == MY_CHARSET_BIN:
            typ = dt.Binary
        else:
            typ = dt.String

    else:
        typ = _type_mapping[typename]

    # projection columns are always nullable
    return typ(nullable=True)


# ported from my_decimal.h:my_decimal_length_to_precision in mariadb
def _decimal_length_to_precision(*, length: int, scale: int, is_unsigned: bool) -> int:
    return length - (scale > 0) - (not (is_unsigned or not length))


_num_types = {1, 2, 3, 4, 5, 8, 9}
_char_types = {15, 249, 250, 251, 252, 253, 254, 255}

_type_codes = {
    0: 'DECIMAL',
    1: 'TINY',
    2: 'SHORT',
    3: 'LONG',
    4: 'FLOAT',
    5: 'DOUBLE',
    6: 'NULL',
    7: 'TIMESTAMP',
    8: 'LONGLONG',
    9: 'INT24',
    10: 'DATE',
    11: 'TIME',
    12: 'DATETIME',
    13: 'YEAR',
    15: 'VARCHAR',
    16: 'BIT',
    245: 'JSON',
    246: 'NEWDECIMAL',
    247: 'ENUM',
    248: 'SET',
    249: 'TINY_BLOB',
    250: 'MEDIUM_BLOB',
    251: 'LONG_BLOB',
    252: 'BLOB',
    253: 'VAR_STRING',
    254: 'STRING',
    255: 'GEOMETRY',
}


_type_mapping = {
    'DECIMAL': dt.Decimal,
    'TINY': dt.Int8,
    'SHORT': dt.Int16,
    'LONG': dt.Int32,
    'FLOAT': dt.Float32,
    'DOUBLE': dt.Float64,
    'NULL': dt.Null,
    'TIMESTAMP': lambda nullable: dt.Timestamp(timezone='UTC', nullable=nullable),
    'LONGLONG': dt.Int64,
    'INT24': dt.Int32,
    'DATE': dt.Date,
    'TIME': dt.Time,
    'DATETIME': dt.Timestamp,
    'YEAR': dt.Int8,
    'VARCHAR': dt.String,
    'JSON': dt.JSON,
    'NEWDECIMAL': dt.Decimal,
    'ENUM': dt.String,
    'SET': lambda nullable: dt.Array(dt.string, nullable=nullable),
    'TINY_BLOB': dt.Binary,
    'MEDIUM_BLOB': dt.Binary,
    'LONG_BLOB': dt.Binary,
    'BLOB': dt.Binary,
    'VAR_STRING': dt.String,
    'STRING': dt.String,
    'GEOMETRY': dt.Geometry,
}


class _FieldFlags:
    """Flags used to disambiguate field types.

    Gaps in the flag numbers are because we do not map in flags that are
    of no use in determining the field's type, such as whether the field
    is a primary key or not.
    """

    UNSIGNED = 1 << 5
    SET = 1 << 11
    NUM = 1 << 15

    __slots__ = ('value',)

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


@dt.dtype.register(SingleStoreDBDialect, (sa.NUMERIC, singlestoredb.NUMERIC))
def sa_singlestoredb_numeric(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    # https://dev.mysql.com/doc/refman/8.0/en/fixed-point-types.html
    return dt.Decimal(satype.precision or 10, satype.scale or 0, nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.YEAR)
@dt.dtype.register(SingleStoreDBDialect, singlestoredb.TINYINT)
def sa_singlestoredb_tinyint(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Int8(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.BIT)
def sa_singlestoredb_bit(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    if 1 <= (length := satype.length) <= 8:
        return dt.Int8(nullable=nullable)
    elif 9 <= length <= 16:
        return dt.Int16(nullable=nullable)
    elif 17 <= length <= 32:
        return dt.Int32(nullable=nullable)
    elif 33 <= length <= 64:
        return dt.Int64(nullable=nullable)
    else:
        raise ValueError(f'Invalid SingleStoreDB BIT length: {length:d}')


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.FLOAT)
def sa_real(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Float32(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.TIMESTAMP)
def sa_singlestoredb_timestamp(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Timestamp(timezone='UTC', nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.DATETIME)
def sa_singlestoredb_datetime(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Timestamp(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.SET)
def sa_singlestoredb_set(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Array(dt.string, nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.DOUBLE)
def sa_singlestoredb_double(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    # TODO: handle asdecimal=True
    return dt.Float64(nullable=nullable)


@dt.dtype.register(
    SingleStoreDBDialect,
    (
        singlestoredb.TINYBLOB,
        singlestoredb.MEDIUMBLOB,
        singlestoredb.BLOB,
        singlestoredb.LONGBLOB,
        singlestoredb.BINARY,
        singlestoredb.VARBINARY,
    ),
)
def sa_binary(_: Any, satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Binary(nullable=nullable)


# TODO(kszucs): unsigned integers


@dt.dtype.register((singlestoredb.DOUBLE, singlestoredb.REAL))
def singlestoredb_double(satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Float64(nullable=nullable)


@dt.dtype.register(singlestoredb.FLOAT)
def singlestoredb_float(satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Float32(nullable=nullable)


@dt.dtype.register(singlestoredb.TINYINT)
def singlestoredb_tinyint(satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Int8(nullable=nullable)


@dt.dtype.register(singlestoredb.BLOB)
def singlestoredb_blob(satype: Any, nullable: bool = True) -> dt.DataType:
    return dt.Binary(nullable=nullable)


class SingleStoreDBDateTime(singlestoredb.DATETIME):
    """Custom DATETIME type for SingleStoreDB that handles zero values."""

    def result_processor(self, *_: Any) -> Callable[..., Any]:
        return lambda v: None if v == '0000-00-00 00:00:00' else v


@dt.dtype.register(SingleStoreDBDateTime)
def singlestoredb_timestamp(_: Any, nullable: bool = True) -> dt.DataType:
    return dt.Timestamp(nullable=nullable)


@to_sqla_type.register(SingleStoreDBDialect, dt.Timestamp)
def _singlestoredb_timestamp(*_: Any) -> dt.DataType:
    return SingleStoreDBDateTime()
