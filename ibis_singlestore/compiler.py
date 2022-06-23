import sqlalchemy.dialects.mysql as singlestore
import toolz

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
)
from ibis.backends.base.sql.alchemy.registry import _geospatial_functions
from .registry import operation_registry


class SingleStoreExprTranslator(AlchemyExprTranslator):
    # https://dev.singlestore.com/doc/refman/8.0/en/spatial-function-reference.html
    _registry = toolz.merge(operation_registry, _geospatial_functions)
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Boolean: singlestore.BOOLEAN,
            dt.Int8: singlestore.TINYINT,
            dt.Int16: singlestore.INTEGER,
            dt.Int32: singlestore.INTEGER,
            dt.Int64: singlestore.BIGINT,
            dt.Float16: singlestore.FLOAT,
            dt.Float32: singlestore.FLOAT,
            dt.Float64: singlestore.DOUBLE,
            dt.String: singlestore.VARCHAR,
        }
    )
    _bool_aggs_need_cast_to_int32 = False


rewrites = SingleStoreExprTranslator.rewrites


class SingleStoreCompiler(AlchemyCompiler):
    translator_class = SingleStoreExprTranslator
