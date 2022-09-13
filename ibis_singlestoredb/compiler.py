from __future__ import annotations

import ibis.expr.datatypes as dt
import sqlalchemy.dialects.mysql as singlestoredb
import toolz
from ibis.backends.base.sql.alchemy import AlchemyCompiler
from ibis.backends.base.sql.alchemy import AlchemyExprTranslator
from ibis.backends.base.sql.alchemy.registry import _geospatial_functions

from .registry import operation_registry


class SingleStoreDBExprTranslator(AlchemyExprTranslator):
    # https://dev.singlestore.com/doc/refman/8.0/en/spatial-function-reference.html
    _registry = toolz.merge(operation_registry, _geospatial_functions)
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Boolean: singlestoredb.BOOLEAN,
            dt.Int8: singlestoredb.TINYINT,
            dt.Int16: singlestoredb.INTEGER,
            dt.Int32: singlestoredb.INTEGER,
            dt.Int64: singlestoredb.BIGINT,
            dt.Float16: singlestoredb.FLOAT,
            dt.Float32: singlestoredb.FLOAT,
            dt.Float64: singlestoredb.DOUBLE,
            dt.String: singlestoredb.VARCHAR,
        },
    )
    _bool_aggs_need_cast_to_int32 = False


rewrites = SingleStoreDBExprTranslator.rewrites


class SingleStoreDBCompiler(AlchemyCompiler):
    translator_class = SingleStoreDBExprTranslator
