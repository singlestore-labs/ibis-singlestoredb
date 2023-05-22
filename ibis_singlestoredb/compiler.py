from __future__ import annotations

from typing import Any

import ibis.expr.datatypes as dt
import sqlalchemy as sa
import sqlalchemy_singlestoredb as singlestoredb
from ibis.backends.base.sql.alchemy import AlchemyCompiler
from ibis.backends.base.sql.alchemy import AlchemyExprTranslator
from ibis.backends.base.sql.alchemy import to_sqla_type

from .registry import operation_registry


class SingleStoreDBExprTranslator(AlchemyExprTranslator):
    # https://dev.mysql.com/doc/refman/8.0/en/spatial-function-reference.html
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _integer_to_timestamp = sa.func.from_unixtime
    native_json_type = False
    _dialect_name = 'singlestoredb'


rewrites = SingleStoreDBExprTranslator.rewrites


class SingleStoreDBCompiler(AlchemyCompiler):
    translator_class = SingleStoreDBExprTranslator
    support_values_syntax_in_select = False


_SINGLESTOREDB_TYPE_MAP = {
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
}


@to_sqla_type.register(singlestoredb.dialect, tuple(_SINGLESTOREDB_TYPE_MAP.keys()))
def _simple_types(_: Any, itype: dt.DataType) -> dt.DataType:
    return _SINGLESTOREDB_TYPE_MAP[type(itype)]
