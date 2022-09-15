"""The SingleStoreDB backend."""
from __future__ import annotations

import atexit
import contextlib
import warnings
from typing import Any
from typing import Dict
from typing import Generator
from typing import Optional

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.dialects.mysql as singlestoredb
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.base.sql.registry.helpers import quote_identifier
from ibis.expr.types import AnyColumn
from pandas.io.json import build_table_schema
from singlestoredb.connection import build_params
from sqlalchemy_singlestoredb.base import SingleStoreDBDialect

from . import functions as fn
from .compiler import SingleStoreDBCompiler
from .datatypes import _type_from_cursor_info


def _series_sqlalchemy_type(
    col: pd.Series,
    dtype: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Determine the SQLAlchemy type for a given pd.Series

    Parameters
    ----------
    col: pd.Series
        The pd.Series to inspect

    dtype: Dict[str, Any], optional
        Dictionary of data type overrides

    Returns
    -------
    SQLAlchemy data type

    """

    if dtype is not None and col.name in dtype:
        return dtype[col.name]

    # NOTE: It's dangerous to import private libraries, but we want to match
    #       their behavior as close as possible.
    import pandas._libs.lib as lib

    # Infer type of column, while ignoring missing values.
    # Needed for inserting typed data containing NULLs, GH 8778.
    col_type = lib.infer_dtype(col, skipna=True)

    import sqlalchemy.types as st

    if col_type == 'datetime64' or col_type == 'datetime':
        # GH 9086: TIMESTAMP is the suggested type if the column contains
        # timezone information
        try:
            if col.dt.tz is not None:
                return st.TIMESTAMP(timezone=True)
        except AttributeError:
            # The column is actually a DatetimeIndex
            # GH 26761 or an Index with date-like data e.g. 9999-01-01
            if getattr(col, 'tz', None) is not None:
                return st.TIMESTAMP(timezone=True)
        return st.DateTime

    if col_type == 'timedelta64':
        warnings.warn(
            "the 'timedelta' type is not supported, and will be "
            'written as integer values (ns frequency) to the database.',
            UserWarning,
            stacklevel=8,
        )
        return st.BigInteger

    elif col_type == 'floating':
        if col.dtype == 'float32':
            return st.Float(precision=23)
        else:
            return st.Float(precision=53)

    elif col_type == 'integer':
        # GH35076 Map pandas integer to optimal SQLAlchemy integer type
        if col.dtype.name.lower() in ('int8', 'uint8', 'int16'):
            return st.SmallInteger
        elif col.dtype.name.lower() in ('uint16', 'int32'):
            return st.Integer
        elif col.dtype.name.lower() == 'uint64':
            raise ValueError('Unsigned 64 bit integer datatype is not supported')
        else:
            return st.BigInteger

    elif col_type == 'boolean':
        return st.Boolean

    elif col_type == 'date':
        return st.Date

    elif col_type == 'time':
        return st.Time

    elif col_type == 'complex':
        raise ValueError('Complex datatypes not supported')

    elif col_type == 'decimal':
        return st.DECIMAL(60, 30)

    return st.Text


def _ibis_schema_to_sqlalchemy_dtypes(df_schema: ibis.Schema) -> Dict[str, Any]:
    """
    Convert an Ibis Schema to a dict of SQLAlchemy types.

    Parameters
    ----------
    schema: ibis.Schema
        Schema object to convert

    Returns
    -------
    Dict[str, Any]

    """
    from ibis.backends.base.sql.alchemy import datatypes
    return dict(
        zip(
            df_schema.names,
            [datatypes.to_sqla_type(x) for x in df_schema.types],
        ),
    )


def _infer_dtypes(
    frame: pd.DataFrame,
    dtype: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Infer the SQLAlchemy dtypes for a DataFrame.

    Parameters
    ----------
    frame : pd.DataFrame
        The DataFrame to inspect

    Returns
    -------
    Dict[str, sa.type]

    """
    return dict([
        (str(frame.columns[i]), _series_sqlalchemy_type(frame.iloc[:, i], dtype))
        for i in range(len(frame.columns))
    ])


class Backend(BaseAlchemyBackend):
    name = 'singlestoredb'
    compiler = SingleStoreDBCompiler

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

        super().do_connect(
            sa.create_engine(
                alchemy_url,
                echo=kwargs.get('echo', False), future=kwargs.get('future', False),
            ),
        )

        self.sync_functions()

#   @contextlib.contextmanager
#   def begin(self) -> Generator[Any, Any, Any]:
#       with super().begin() as bind:
#           previous_timezone = bind.execute(
#               'SELECT @@session.time_zone',
#           ).scalar()
#           unset_timezone = False
#           try:
#               bind.execute("SET @@session.time_zone = 'UTC'")
#               unset_timezone = True
#           except Exception as e:
#               warnings.warn(f"Couldn't set singlestore timezone: {str(e)}")

#           try:
#               yield bind
#           finally:
#               if unset_timezone:
#                   query = "SET @@session.time_zone = '{}'"
#                   bind.execute(query.format(previous_timezone))

    def create_table(
        self,
        name: str,
        expr: pd.DataFrame | ir.TableExpr | None = None,
        schema: sch.Schema | None = None,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Create a new table.
        Parameters
        ----------
        name
            Name of the new table.
        expr
            An Ibis table expression or pandas DataFrame that will be used to
            extract the schema and the data of the new table. If not provided,
            `schema` must be given.
        schema
            The schema for the new table. Only one of `schema` or `expr` can be
            provided.
        database
            Name of the database where the table will be created, if not the
            default.
        force
            Check whether a table exists before creating it
        """
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

            if schema is not None:
                dtype = _ibis_schema_to_sqlalchemy_dtypes(schema)
            else:
                dtype = _infer_dtypes(expr)

            expr.to_sql(
                name,
                self.con,
                index=False,
                if_exists='replace' if force else 'fail',
                dtype=dtype,
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
            t = self._table_from_schema(
                name, schema, database=database or self.current_database,
            )

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

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Infer the schema of `query`."""
        result = self.con.execute(f'SELECT * FROM ({query}) _ LIMIT 0')
        cursor = result.cursor._cursor
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
        return f'CREATE OR REPLACE VIEW {name} AS {definition}'

    def _register_temp_view_cleanup(self, name: str, raw_name: str) -> None:
        query = f'DROP VIEW IF EXISTS {name}'

        def drop(self: Backend, raw_name: str, query: str) -> None:
            self.con.execute(query)
            self._temp_views.discard(raw_name)

        atexit.register(drop, self, raw_name, query)


# TODO(kszucs): unsigned integers


@dt.dtype.register((singlestoredb.DOUBLE, singlestoredb.REAL))
def singlestoredb_double(satype: Any, nullable: bool = True) -> dt.Float64:
    return dt.Float64(nullable=nullable)


@dt.dtype.register(singlestoredb.FLOAT)
def singlestoredb_float(satype: Any, nullable: bool = True) -> dt.Float32:
    return dt.Float32(nullable=nullable)


@dt.dtype.register(singlestoredb.TINYINT)
def singlestoredb_tinyint(satype: Any, nullable: bool = True) -> dt.Int8:
    return dt.Int8(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.YEAR)
def singlestoredb_year(dialect: Any, satype: Any, nullable: bool = True) -> dt.Int16:
    return dt.Int16(nullable=nullable)


@dt.dtype.register(singlestoredb.BLOB)
def singlestoredb_blob(satype: Any, nullable: bool = True) -> dt.Binary:
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.BIT)
def singlestoredb_bit(dialect: Any, satype: Any, nullable: bool = True) -> dt.Binary:
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.BINARY)
def singlestoredb_binary(dialect: Any, satype: Any, nullable: bool = True) -> dt.Binary:
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.VARBINARY)
def singlestoredb_varbinary(
    dialect: Any,
    satype: Any,
    nullable: bool = True,
) -> dt.Binary:
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.LONGBLOB)
def singlestoredb_longblob(dialect: Any, satype: Any, nullable: bool = True) -> dt.Binary:
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.MEDIUMBLOB)
def singlestoredb_mediumblob(
    dialect: Any,
    satype: Any,
    nullable: bool = True,
) -> dt.Binary:
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.TINYBLOB)
def singlestoredb_tinyblob(dialect: Any, satype: Any, nullable: bool = True) -> dt.Binary:
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SingleStoreDBDialect, singlestoredb.JSON)
def singlestoredb_json(dialect: Any, satype: Any, nullable: bool = True) -> dt.JSON:
    return dt.JSON(nullable=nullable)
