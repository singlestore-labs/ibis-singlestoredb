"""The SingleStoreDB backend."""
from __future__ import annotations

import re
import warnings
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Union

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util
import pandas as pd
import sqlalchemy as sa
import sqlalchemy_singlestoredb as singlestoredb
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from singlestoredb.connection import build_params

from . import functions as fn
from .compiler import SingleStoreDBCompiler
from .database import SingleStoreDBDatabase
from .database import SingleStoreDBDatabaseTable
from .datatypes import _type_from_cursor_info
from .datatypes import SingleStoreDBDateTime
from .expr import SingleStoreDBTable
# from . import functions as fn


__version__ = '0.3.0'


class Backend(BaseAlchemyBackend):
    name = 'singlestoredb'

    compiler = SingleStoreDBCompiler
    database_class = SingleStoreDBDatabase
    table_class = SingleStoreDBDatabaseTable
    table_expr_class = SingleStoreDBTable
    supports_create_or_replace = False

    _database_name: Optional[str] = None

    @property
    def database_name(self) -> Optional[str]:
        """Get the currently selected database."""
        return self._database_name

    @database_name.setter
    def database_name(self, value: Optional[str]) -> None:
        """Set the default database name."""
        # TODO: unset database
        if value is None:
            return

        # TODO: escape value
        value = str(value)
        if self._database_name != value and hasattr(self, 'con'):
            self.raw_sql(f'use {value}')

        self._database_name = value

    def create_database(self, name: str, force: bool = False) -> None:
        """
        Create a new database.
        Parameters
        ----------
        name : str
            Name for the new database
        force : bool, optional
            If `True`, an exception is raised if the database already exists.
        """
        if force and name.lower() in [x.lower() for x in self.list_databases()]:
            raise ValueError(f'Database with the name "{name}" already exists.')
        # TODO: escape name
        self.raw_sql(f'CREATE DATABASE IF NOT EXISTS {name}')

    @property
    def show(self) -> Any:
        """Access to SHOW commands on the server."""
        return self.con.raw_connection().show

    @property
    def globals(self) -> Any:
        """Accessor for global variables in the server."""
        return self.con.raw_connection().globals

    @property
    def locals(self) -> Any:
        """Accessor for local variables in the server."""
        return self.con.raw_connection().locals

    @property
    def cluster_globals(self) -> Any:
        """Accessor for cluster global variables in the server."""
        return self.con.raw_connection().cluster_globals

    @property
    def cluster_locals(self) -> Any:
        """Accessor for cluster local variables in the server."""
        return self.con.raw_connection().cluster_locals

    @property
    def vars(self) -> Any:
        """Accessor for variables in the server."""
        return self.con.raw_connection().vars

    @property
    def cluster_vars(self) -> Any:
        """Accessor for cluster variables in the server."""
        return self.con.raw_connection().cluster_vars

    def sync_functions(self) -> None:
        """Synchronize client APIs with server functions."""
        for row in self.raw_sql('SHOW FUNCTIONS'):
            fn.build_function(self, row[0])

    def do_connect(self, *args: str, **kwargs: Any) -> None:
        """Connect to a SingleStoreDB database."""
        if args:
            params = build_params(host=args[0], **kwargs)
        else:
            params = build_params(**kwargs)

        driver = params.pop('driver', None)
        if driver and not driver.startswith('singlestoredb+'):
            driver = 'singlestoredb+{}'.format(driver)

        alchemy_url = self._build_alchemy_url(
            url=params.pop('url', None),
            host=params.pop('host', None),
            port=params.pop('port', None),
            user=params.pop('user', None),
            password=params.pop('password', None),
            database=params.pop('database', None),
            driver=driver,
        )

        alchemy_url.set(query={k: str(v) for k, v in params.items()})

        self.database_name = alchemy_url.database

        engine = sa.create_engine(alchemy_url, poolclass=sa.pool.StaticPool)

        @sa.event.listens_for(engine, 'connect')
        def connect(
            dbapi_connection: singlestoredb.Connection,
            connection_record: Any,
        ) -> None:
            with dbapi_connection.cursor() as cur:
                try:
                    cur.execute("SET @@session.time_zone = 'UTC'")
                except sa.exc.OperationalError:
                    warnings.warn('Unable to set session timezone to UTC.')

        super().do_connect(engine)

        self.sync_functions()

    def _table_from_schema(
        self,
        name: str,
        schema: sch.Schema,
        temp: bool = False,
        database: Optional[str] = None,
        **kwargs: Any,
    ) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        prefixes = []
        if temp:
            prefixes.append(self._temporary_prefix)
        storage_type = kwargs.pop('storage_type', None)
        if storage_type:
            prefixes.append(storage_type.upper())
        return sa.Table(
            name,
            sa.MetaData(),
            *columns,
            prefixes=prefixes,
            quote=self.compiler.translator_class._quote_table_names,
            **kwargs,
        )

    def _merge_schema_overrides(
        self,
        schema: sch.Schema,
        overrides: Union[Dict[str, str], sch.Schema],
    ) -> sch.Schema:
        """
        Merge type overrides into a schema.

        Parameters
        ----------
        schema : Schema
            The starting schema
        overrides : dict or Schema
            The override types

        Returns
        -------
        Schema

        """
        if not isinstance(overrides, sch.Schema):
            overrides = sch.Schema.from_tuples(list(overrides.items()))

        items = list(schema.items())
        for i, item in enumerate(items):
            if item[0] in overrides:
                items[i] = (item[0], overrides[item[0]])

        return ibis.Schema.from_tuples(items)

    def create_table(
        self,
        name: str,
        expr: Optional[Union[pd.DataFrame, ir.TableExpr]] = None,
        schema: Optional[sch.Schema] = None,
        database: Optional[str] = None,
        force: bool = False,
        storage_type: Optional[str] = None,
        schema_overrides: Optional[Union[Dict[str, str], sch.Schema]] = None,
    ) -> ir.Table:
        """
        Create a new table.

        Parameters
        ----------
        name : str
            Name of the new table.
        expr : Schema or DataFrame, optional
            An Ibis table expression or pandas DataFrame that will be used to
            extract the schema and the data of the new table. If not provided,
            ``schema`` must be given.
        schema : Schema, optional
            The schema for the new table. Only one of ``schema`` or ``expr`` can be
            provided.
        database : str, optional
            Name of the database where the table will be created, if not the
            default.
        force : bool, optional
            Check whether a table exists before creating it
        storage_type : str, optional
            The storage type of table to create: COLUMNSTORE or ROWSTORE
        schema_overrides : dict or Schema, optional
            If the ``expr`` is a DataFrame, the data types of specific fields
            can be overridden using a partial Schema or dict with keys for
            the overridden columns. Values of the dict are simply the string
            names of the data types: bool, int32, int64, float64, etc.

        Returns
        -------
        Table expression

        """
        if storage_type:
            storage_type = storage_type.upper()
            if storage_type not in ['ROWSTORE', 'COLUMNSTORE']:
                raise ValueError(f'Unknown table type: {storage_type}')
            if storage_type == 'COLUMNSTORE':
                storage_type = None

        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                'Creating tables from a different database is not yet '
                'implemented',
            )

        if expr is None and schema is None:
            raise ValueError('You must pass either an expression or a schema')

        drop = False
        if name.lower() in [x.lower() for x in self.list_tables()]:
            if force:
                drop = True
            else:
                raise ValueError(
                    f'Table `{name}` already exists. '
                    'Use force=True to overwrite.',
                )

        if isinstance(expr, pd.DataFrame):
            if schema is not None:
                pd_schema_names = ibis.pandas.connect(
                    {name: expr},
                ).table(name).schema().names
                if not sorted(pd_schema_names) == sorted(sch.schema(schema).names):
                    raise TypeError(
                        'Expression schema is not equal to passed schema. '
                        'Try passing the expression without the schema',
                    )

            # TODO: Should this be done in `insert` as well?
            expr = expr.copy()
            for column in expr:
                try:
                    expr[column].dt.tz_localize('UTC')
                except (AttributeError, TypeError):
                    pass

            if schema is None:
                schema = ibis.pandas.connect({name: expr}).table(name).schema()

            if schema_overrides:
                schema = self._merge_schema_overrides(schema, schema_overrides)

            self._schemas[self._fully_qualified_name(name, database)] = schema

            if drop:
                self.drop_table(name, force=True)

            t = self._table_from_schema(
                name, schema, database=database or self.current_database,
                storage_type=storage_type,
            )

            if ibis.options.verbose:
                from sqlalchemy.schema import CreateTable
                create_stmt = CreateTable(t).compile(self.con.engine)
                ibis.util.log(str(create_stmt).strip())

            with self.begin() as bind:
                t.create(bind=bind, checkfirst=force)
                expr.to_sql(
                    name,
                    self.con,
                    index=False,
                    if_exists='append',
                )

        elif isinstance(expr, ir.TableExpr) or schema is not None:
            if expr is not None and schema is not None:
                if not sorted(expr.schema().names) == sorted(sch.schema(schema).names):
                    raise TypeError(
                        'Expression schema is not equal to passed schema. '
                        'Try passing the expression without the schema',
                    )

            if schema is None:
                schema = expr.schema()  # type: ignore

            self._schemas[self._fully_qualified_name(name, database)] = schema

            if drop:
                self.drop_table(name, force=True)

            t = self._table_from_schema(
                name, schema, database=database or self.current_database,
                storage_type=storage_type,
            )

            if ibis.options.verbose:
                from sqlalchemy.schema import CreateTable
                create_stmt = CreateTable(t).compile(self.con.engine)
                ibis.util.log(str(create_stmt).strip())

            with self.begin() as bind:
                t.create(bind=bind, checkfirst=force)
                if expr is not None:
                    bind.execute(
                        t.insert().from_select(list(expr.columns), expr.compile()),
                    )

        else:
            raise TypeError(
                '`expr` and/or `schema` are not an expected type: {} / {}'.format(
                    type(expr).__name__, type(schema).__name__,
                ),
            )

        return self.table(name)

    @staticmethod
    def _new_sa_metadata() -> sa.MetaData:
        meta = sa.MetaData()

        @sa.event.listens_for(meta, 'column_reflect')
        def column_reflect(
            inspector: Any,
            table: Any,
            column_info: Dict[str, Any],
        ) -> None:
            if isinstance(column_info['type'], singlestoredb.DATETIME):
                column_info['type'] = SingleStoreDBDateTime()

        return meta

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        if (
            re.search(r'^\s*SELECT\s', query, flags=re.MULTILINE | re.IGNORECASE)
            is not None
        ):
            query = f'({query})'

        with self.begin() as con:
            result = con.exec_driver_sql(f'SELECT * FROM {query} _ LIMIT 0')
            cursor = result.cursor
            yield from (
                (field.name, _type_from_cursor_info(descr, field))
                for descr, field in zip(cursor.description, cursor._result.fields)
            )

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Infer the schema of `query`."""
        with self.begin() as con:
            result = con.exec_driver_sql(f'SELECT * FROM ({query}) _ LIMIT 0')
            cursor = result.cursor
            fields = [
                (field.name, _type_from_cursor_info(descr, field))
                for descr, field in zip(cursor.description, cursor._result.fields)
            ]
            return sch.Schema.from_tuples(fields)

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        return f'CREATE VIEW {name} AS {definition}'
