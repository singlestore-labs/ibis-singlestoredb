from __future__ import annotations

from ibis.backends.base.sql.alchemy import AlchemyDatabase
from ibis.backends.base.sql.alchemy import AlchemyTable


class SingleStoreDBDatabase(AlchemyDatabase):
    """SingleStoreDB database class."""


class SingleStoreDBDatabaseTable(AlchemyTable):
    """SingleStoreDB table class."""
