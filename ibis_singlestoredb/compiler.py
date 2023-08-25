from __future__ import annotations

import sqlalchemy as sa
from ibis.backends.base.sql.alchemy import AlchemyCompiler
from ibis.backends.base.sql.alchemy import AlchemyExprTranslator

from .datatypes import SingleStoreDBType
from .registry import operation_registry


class SingleStoreDBExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _integer_to_timestamp = sa.func.from_unixtime
    native_json_type = False
    _dialect_name = 'singlestoredb'
    type_mapper = SingleStoreDBType


rewrites = SingleStoreDBExprTranslator.rewrites


class SingleStoreDBCompiler(AlchemyCompiler):
    translator_class = SingleStoreDBExprTranslator
    support_values_syntax_in_select = False
