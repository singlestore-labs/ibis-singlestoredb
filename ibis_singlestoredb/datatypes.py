from __future__ import annotations

from functools import partial
from typing import Any
from typing import Tuple

import ibis.expr.datatypes as dt
import sqlalchemy.types as sat
import sqlalchemy_singlestoredb as singlestoredb
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sql.alchemy.datatypes import UUID
from ibis.common.exceptions import UnsupportedOperationError

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

    kwargs = {}

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

    elif typename == 'TIMESTAMP':
        typ = dt.Timestamp
        kwargs['timezone'] = 'UTC'
        # if scale > 0:
        #     kwargs['scale'] = scale

    elif typename == 'DATETIME':
        typ = dt.Timestamp
        # if scale > 0:
        #     kwargs['scale'] = scale

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
    return typ(nullable=True, **kwargs)


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


class SingleStoreDBDateTime(singlestoredb.DATETIME):
    """Custom DATETIME type for SingleStoreDB that handles zero values."""

    def result_processor(self, *_: Any) -> Any:
        return lambda v: None if v == '0000-00-00 00:00:00' else v


_to_singlestoredb_types = {
    dt.Boolean: singlestoredb.BOOLEAN,
    dt.Int8: singlestoredb.TINYINT,
    dt.Int16: singlestoredb.SMALLINT,
    dt.Int32: singlestoredb.INTEGER,
    dt.Int64: singlestoredb.BIGINT,
    dt.Float16: singlestoredb.FLOAT,
    dt.Float32: singlestoredb.FLOAT,
    dt.Float64: singlestoredb.DOUBLE,
    dt.String: singlestoredb.TEXT,
    dt.JSON: singlestoredb.JSON,
    dt.Timestamp: SingleStoreDBDateTime,
}

_from_singlestoredb_types = {
    singlestoredb.BIGINT: dt.Int64,
    singlestoredb.BINARY: dt.Binary,
    singlestoredb.BLOB: dt.Binary,
    singlestoredb.BOOLEAN: dt.Boolean,
    singlestoredb.DATETIME: dt.Timestamp,
    singlestoredb.DOUBLE: dt.Float64,
    singlestoredb.FLOAT: dt.Float32,
    singlestoredb.INTEGER: dt.Int32,
    singlestoredb.JSON: dt.JSON,
    singlestoredb.LONGBLOB: dt.Binary,
    singlestoredb.LONGTEXT: dt.String,
    singlestoredb.MEDIUMBLOB: dt.Binary,
    singlestoredb.MEDIUMINT: dt.Int32,
    singlestoredb.MEDIUMTEXT: dt.String,
    singlestoredb.REAL: dt.Float64,
    singlestoredb.SMALLINT: dt.Int16,
    singlestoredb.TEXT: dt.String,
    singlestoredb.DATE: dt.Date,
    singlestoredb.TINYBLOB: dt.Binary,
    singlestoredb.TINYINT: dt.Int8,
    singlestoredb.TINYTEXT: dt.String,
    singlestoredb.VARBINARY: dt.Binary,
    singlestoredb.VARCHAR: dt.String,
    singlestoredb.ENUM: dt.String,
    singlestoredb.CHAR: dt.String,
    singlestoredb.TIME: dt.Time,
    singlestoredb.YEAR: dt.UInt8,
    SingleStoreDBDateTime: dt.Timestamp,
    UUID: dt.String,
}

_unsigned_int_map = {
    dt.Int8: dt.UInt8,
    dt.Int16: dt.UInt16,
    dt.Int32: dt.UInt32,
    dt.Int64: dt.UInt64,
}


class SingleStoreDBType(AlchemyType):

    dialect = 'singlestoredb'

    @classmethod
    def from_ibis(cls, dtype: Any) -> Any:
        try:
            out = _to_singlestoredb_types[type(dtype)]
            if isinstance(dtype, dt.Timestamp):
                if getattr(dtype, 'timezone', 'UTC') not in [False, None, 'UTC', 'utc']:
                    raise UnsupportedOperationError('All timestamps must be in UTC')
                scale = getattr(dtype, 'scale', 0)
                if scale:
                    out = out(fsp=scale)
                else:
                    out = out()
            elif isinstance(dtype, dt.Decimal):
                out = out(
                    precision=getattr(dtype, 'precision', None) or None,
                    scale=getattr(dtype, 'scale', None) or None,
                )
            elif isinstance(dtype, dt.UnsignedInteger):
                out = out(unsigned=True)
            return out
        except KeyError:
            return super().from_ibis(dtype)

    @classmethod
    def to_ibis(cls, typ: Any, nullable: bool = True) -> Any:
        if isinstance(typ, (sat.NUMERIC, singlestoredb.NUMERIC, singlestoredb.DECIMAL)):
            return dt.Decimal(typ.precision or 10, typ.scale or 0, nullable=nullable)
        elif isinstance(typ, singlestoredb.BIT):
            return dt.Int64(nullable=nullable)
        elif isinstance(
            typ, (singlestoredb.TIMESTAMP, singlestoredb.DATETIME, SingleStoreDBDateTime),
        ):
            if getattr(typ, 'timezone', 'UTC') not in [False, None, 'UTC', 'utc']:
                raise UnsupportedOperationError('All timestamps must be in UTC')
            kwargs = dict(
                # scale=getattr(typ, 'scale', None),
                timezone=None if isinstance(typ, singlestoredb.DATETIME) else 'UTC',
            )
            return dt.Timestamp(nullable=nullable, **kwargs)
        elif isinstance(typ, singlestoredb.SET):
            return dt.Set(dt.string, nullable=nullable)
        elif dtype := _from_singlestoredb_types[type(typ)]:
            if getattr(typ, 'unsigned', False):
                dtype = _unsigned_int_map.get(dtype, dtype)
            return dtype(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)
