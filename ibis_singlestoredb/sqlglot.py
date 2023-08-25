from __future__ import annotations

from sqlglot.dialects.dialect import Dialect
from sqlglot.dialects.mysql import MySQL


# Define a sqlglot dialect based on MySQL if a real one doesn't exist
if Dialect.get('singlestoredb') is None:

    class SingleStoreDB(MySQL):
        pass


dialect = Dialect['singlestoredb']
