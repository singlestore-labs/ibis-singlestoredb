from __future__ import annotations

from ibis.backends.base.sql.alchemy import AlchemyDatabase
from ibis.backends.base.sql.alchemy import AlchemyTable


class SingleStoreDBDatabase(AlchemyDatabase):
    """SingleStoreDB database class."""
    pass


class SingleStoreDBTable(AlchemyTable):
    """SingleStoreDB table class."""
    pass
