#!/usr/bin/env python
"""
Ibis backend for SingleStore.

Examples
--------
>>> import ibis
>>> conn = ibis.singlestore.connect('user:password@localhost:3306/mydb')
>>> mytbl = conn.table('mytbl')
>>> mytbl[mytbl.name.like('Smith')].execute()

"""
from __future__ import annotations

import contextlib
import re
import warnings
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import sqlalchemy
import sqlalchemy.dialects.mysql as singlestore
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.base.sql.registry.helpers import quote_identifier
from ibis.expr.types import AnyColumn

from . import ddl
from .compiler import SingleStoreCompiler
from .udf import SingleStoreUDA
from .udf import SingleStoreUDF
from .udf import wrap_udf


# TODO: Patch in an `apply` method for demo

def apply(
    self: AnyColumn,
    func: Callable[..., Any],
    axis: int = 0,
    raw: bool = False,
    result_type: Optional[object] = None,
    args: Optional[Tuple[Any, ...]] = None,
    **kwargs: Any,
) -> ir.Expr:
    """
    Apply a database function to a table column.

    Parameters
    ----------
    self : AnyColumn
        Table column to apply the function to
    func : SingleStoreUDF
        Function to apply
    axis : int, optional
        Not supported
    raw : bool, optional
        Not supported
    result_type : type, optional
        Result type of function
    args : tuple, optional
        Additional arguments to function
    **kwargs : keyword-arguments, optional
        Additional keyword arguments to function

    Returns
    -------
    ir.Expr : value expression

    """
    args = args or tuple()
    # name = func.__name__
    func = self.op().table.op().source.create_function(func)
    out = func(self, *args)
    # self.op().table.op().source.raw_sql(f'drop function {name}')
    return out


AnyColumn.apply = apply


class FuncParam(object):
    """
    Function argument definition.

    Parameters
    ----------
    name : str
        Name of the argument
    dtype : str
        Data type of the argument
    default : Any, optional
        Default value
    is_nullable : bool, optional
        Can the value be NULL?
    collate : str, optional
        Collation order

    """

    def __init__(
        self,
        name: str,
        dtype: str,
        default: Any = None,
        is_nullable: bool = False,
        collate: Optional[str] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.default = default
        self.is_nullable = is_nullable
        self.collate = collate


class FuncReturns(object):
    """
    Function output definition.

    Parameters
    ----------
    dtype : str
        Output data type
    is_nullable : bool, optional
        Can the value be NULL?
    collate : str, optional
        Collation order

    """

    def __init__(
        self,
        dtype: str,
        is_nullable: bool = False,
        collate: Optional[str] = None,
    ):
        self.dtype = dtype
        self.is_nullable = is_nullable
        self.collate = collate


param = FuncParam
ret = FuncReturns


class FuncDict(Dict[str, Union[SingleStoreUDF, SingleStoreUDA]]):
    """
    Accessor for holding UDFs and UDAs.

    This object is not instantiated directly. It is accessed through
    the `funcs` attribute of the backend object.

    Parameters
    ----------
    con : Backend
        Backend object associated with the functions

    """

    _con: Backend
    _database_name: str

    _db2py_type: Dict[str, str] = {
        'int64': 'int',
        'bigint': 'int',
        'string': 'str',
        'varchar': 'str',
        'text': 'str',
    }

    def __init__(self, con: Backend):
        super(dict, self).__init__()
        self.__dict__['_con'] = con
        self.__dict__['_database_name'] = con.database_name
        self._refresh()

    def _get_py_type(self, typ: Optional[str]) -> str:
        """
        Return the Python type for a given database type.

        Parameters
        ----------
        typ : str
            Name of the database type

        Returns
        -------
        str

        """
        if typ is None:
            return 'None'
        return type(self)._db2py_type.get(typ, typ)

    def __call__(self, refresh: bool = False) -> FuncDict:
        """
        Apply operations to the function dictionary.

        Parameters
        ----------
        refresh : bool, optional
            Refresh the list of available functions?

        Returns
        -------
        self

        """
        if refresh:
            self._refresh()
        return self

    def _refresh(self) -> None:
        """
        Update functions in the dictionary.

        Returns
        -------
        None

        """
        self.clear()
        db = quote_identifier(self._database_name)
        for item in self._con.raw_sql(f'show functions in {db}').fetchall():
            self[item[0]] = self._make_func(item[0])

    def _has_function(self, name: str) -> bool:
        """
        Indicate whether the function exists in the database or not.

        Parameters
        ----------
        name : str
            Name of the function in question

        Returns
        -------
        bool

        """
        db = quote_identifier(self._database_name)
        # qname = quote_identifier(item[0])
        funcs = self._con.raw_sql(f'show functions in {db} like {name}').fetchall()
        if len(funcs) == 1:
            return True
        if len(funcs) == 0:
            return False
        raise ValueError(
            'More than one function matches name: {}'.format(', '.join(funcs)),
        )

    def _make_func(self, name: str) -> Union[SingleStoreUDF, SingleStoreUDA]:
        """
        Create a Python wrapper for the requested UDF / UDA.

        Parameters
        ----------
        name : str
            Name of the function

        Returns
        -------
        SingleStoreUDF | SingleStoreUDA

        """
        db = quote_identifier(self._database_name)
        qname = quote_identifier(name)
        proto = self._con.raw_sql(f'show create function {db}.{qname}').fetchall()[0][2]
        proto = re.split(r'\bfunction\s+', proto, flags=re.I)[-1]
        name, proto = proto.split('(', 1)

        if re.search(r'\)\s+returns\s+', proto, flags=re.I):
            sig, ret = re.split(r'\)\s+returns\s+', proto, flags=re.I)
            ret, ftype = re.split(r'\s+as\s+', ret, flags=re.I)
        else:
            ret = None
            sig, ftype = re.split(r'\s+as\s+', proto, flags=re.I)

        ftype, info = ftype.split("'", 1)
        ftype = ftype.strip()

        m = re.search(r"^(.*)'\s+format\s+(\w+)\s*;\s*$", info, flags=re.I)
        if m is None:
            raise ValueError(f'Could not extract code from: {info}')

        code = m.group(1)
        format = m.group(2)
        if name.startswith('`'):
            name = name[1:-1]

        input_names = []
        inputs = []
        for x in sig.split(','):
            m = re.match(r'^\s*(\w+)\s+(\w+)', x)
            if m is None:
                raise ValueError(f'Could not extract parameter names from: {sig}')
            input_names.append(m.group(1))
            inputs.append(m.group(2))

        nullable = [
            not re.search(r'\bnot\s+null\b', x, flags=re.I)
            for x in sig.split(',')
        ]

        inputs = [dict(bigint='int64', text='string')[x] for x in inputs]

        out_nullable = False
        output = ret
        if output:
            out_nullable = not re.search(r'\bnot\s+null\b', output, flags=re.I)
            m = re.match(r'^\s*(\w+)', output)
            if m is None:
                raise ValueError(f'Could not extract nullable information from: {output}')
            output = dict(bigint='int64', text='string')[m.group(1)]

        func_type = type(
            f'UDF_{name}', (SingleStoreUDF,),
            {
                '__doc__': f'{name} function.',
                '__call__': self._make___call__(
                    name, ftype.lower(),
                    list(
                        zip(input_names, inputs, nullable),
                    ),
                    (output, out_nullable), code, format,
                ),
            },
        )
        func = func_type(inputs, output, name)
        func.register(name, self._database_name)
        return func

    def _make___call__(
        self,
        name: str,
        ftype: str,
        inputs: Sequence[Tuple[str, str, bool]],
        output: Optional[Tuple[str, bool]],
        code: str,
        format: str,
    ) -> str:
        """
        Create __call__ method of function.

        Parameters
        ----------
        name : str
            Name of the function
        ftype : str
            Type of the function in the database
        inputs : Sequence[Tuple[str, str, bool]]
            Sequence of (name, type, is_nullable) elements describing
            the inputs of the function
        output : Tuple[str, bool], optional
            Tuple of the form (type, is_nullable) for the return value
            of the function
        code : str
            Code of the UDF / UDA
        format : str
            UDF / UDA output format

        Returns
        -------
        function

        """
        def annotate(typ: Optional[str], nullable: bool) -> str:
            """Generate type annotation."""
            if typ is None:
                return 'None'
            typ = self._get_py_type(typ)
            if nullable:
                return f'Optional[{typ}]'
            return typ

        names = [x[0] for x in inputs]
        types = [annotate(x[1], x[2]) for x in inputs]
        sig = ', '.join([f'{x}: {y}' for x, y in zip(names, types)])
        args = ', '.join(names)
        ret = annotate(*(output or [None, True]))

        new_func = f'def __call__(self, {sig}) -> {ret}:\n' + \
                   f'    return SingleStoreUDF.__call__(self, {args})'
        new_func_code = compile(new_func, '<string>', 'exec')

        loc: Dict[str, Any] = {}
        exec(new_func_code, globals(), loc)

        __call__ = loc['__call__']
        __call__.__doc__ = self._make_func_doc(
            name, ftype, inputs, output, code, format,
        )
        return __call__

    def _make_func_doc(
        self,
        name: str,
        ftype: str,
        inputs: Sequence[Tuple[str, str, bool]],
        output: Optional[Tuple[str, bool]],
        code: str,
        format: str,
    ) -> str:
        """
        Construct the docstring using the function information.

        Parameters
        ----------
        name : str
            Name of the function
        ftype : str
            Type of the function in the database
        inputs : Sequence[Tuple[str, str, bool]]
            Sequence of (name, type, is_nullable) elements describing
            the inputs of the function
        output : Tuple[str, bool], optional
            Tuple of the form (type, is_nullable) for the return value
            of the function
        code : str
            Code of the UDF / UDA
        format : str
            UDF / UDA output format

        Returns
        -------
        str

        """
        doc = [f'Call `{name}` {ftype} function.', '']
        if ftype == 'remote service':
            doc.append(f'Accesses remote service at {code} using {format} format.')
            doc.append('')
        doc.extend(['Parameters', '----------'])
        for name, dtype, nullable in inputs:
            dtype = self._get_py_type(dtype)
            arg = f'{name} : {dtype}'
            if nullable:
                arg += ' or None'
            doc.append(arg)
        if output and output[0]:
            doc.append('')
            doc.extend(['Returns', '-------'])
            ret = '{}'.format(self._get_py_type(output[0]))
            if output[1]:
                ret += ' or None'
            doc.append(ret)
        doc.append('')
        return '\n'.join(doc)

    def __getattr__(self, name: str) -> Union[SingleStoreUDF, SingleStoreUDA]:
        """
        Retrieve the specified attribute.

        Parameters
        ----------
        name : str
            Name of the function

        Returns
        -------
        SingleStoreUDF | SingleStoreUDA

        """
        try:
            return self[name]
        except KeyError:
            if self._has_function(name):
                func = self._make_func(name)
                self[name] = func
                return func
            raise AttributeError(f"'dict' object has no attribute '{name}'")

    def __getitem__(self, name: str) -> Union[SingleStoreUDF, SingleStoreUDA]:
        """
        Retrieve the specified key.

        Parameters
        ----------
        name : str
            Name of the function

        Returns
        -------
        SingleStoreUDF | SingleStoreUDA

        """
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            if self._has_function(name):
                func = self._make_func(name)
                self[name] = func
                return func
            raise

    def __setattr__(
        self,
        name: str,
        value: Union[SingleStoreUDF, SingleStoreUDA],
    ) -> None:
        """
        Set a function in the dictionary.

        Parameters
        ----------
        name : str
            Name of the function
        value : SingleStoreUDF or SingleStoreUDA
            Function value

        Returns
        -------
        None

        """
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """
        Remove an entry from the dictionary.

        Parameters
        ----------
        name : str
            Name of the function

        Returns
        -------
        None

        """
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


class TableAccessor(object):
    """
    Accessor for database table objects.

    Parameters
    ----------
    backend : Backend
        The backend to use for table lookups

    """

    def __init__(self, backend: Backend):
        self._backend = backend

    def __getattr__(self, name: str) -> ir.TableExpr:
        """
        Retrieve the given table.

        Parameters
        ----------
        name : str
            Name of the table

        Returns
        -------
        ir.TableExpr

        """
        return self._backend._table(name)

    def __call__(
        self,
        name: str,
        database: Optional[str] = None,
        schema: Optional[sch.Schema] = None,
    ) -> ir.TableExpr:
        """
        Retrieve the requested table object.

        Parameters
        ----------
        name : str
            Name of the table
        database : str, optional
            Database for the table
        schema : sch.Schema, optional
            Schema of the table

        Returns
        -------
        ir.TableExpr

        """
        return self._backend._table(name, database=database, schema=schema)


class Backend(BaseAlchemyBackend):
    """Ibis backend for SingleStore."""

    name = 'singlestore'
    compiler = SingleStoreCompiler

    _funcs: FuncDict
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

    def do_connect(
        self,
        url: Optional[str] = None,
        host: Optional[str] = 'localhost',
        user: Optional[str] = None,
        password: Optional[str] = None,
        port: Optional[int] = 3306,
        database: Optional[str] = None,
        driver: Optional[str] = None,
    ) -> None:
        """
        Connect to a SingleStore database.

        Parameters
        ----------
        url : str, optional
            Full URL of the database connection in SQLAlchemy form
        host : str, optional
            Name or IP address of the database host
        user : str, optional
            User name for the database connection
        password : str, optional
            Password for the database connection
        port : int, optional
            Port number of the database server
        database : str, optional
            Name of the database to connect to
        driver : str, optional
            Name of the SingleStore database connection driver

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> url = os.environ.get('IBIS_TEST_SINGLESTORE_URL')
        >>> host = os.environ.get('IBIS_TEST_SINGLESTORE_HOST', 'localhost')
        >>> port = int(os.environ.get('IBIS_TEST_SINGLESTORE_PORT', 3306))
        >>> user = os.environ.get('IBIS_TEST_SINGLESTORE_USER', getpass.getuser())
        >>> password = os.environ.get('IBIS_TEST_SINGLESTORE_PASSWORD')
        >>> database = os.environ.get('IBIS_TEST_SINGLESTORE_DATABASE',
        ...                           'ibis_testing')
        >>> con = connect(
        ...     url=url,
        ...     host=host,
        ...     port=port,
        ...     user=user,
        ...     password=password,
        ...     database=database
        ... )
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table('functional_alltypes')
        >>> t
        SingleStoreTable[table]
          name: functional_alltypes
          schema:
            index : int64
            Unnamed: 0 : int64
            id : int32
            bool_col : int8
            tinyint_col : int8
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32
        """
        if url:
            if '//' not in url:
                if driver:
                    url = f'singlestore+{driver}://{url}'
                else:
                    url = f'singlestore://{url}'
            elif not url.startswith('singlestore'):
                url = f'singlestore+{url}'
            driver = None
        elif driver:
            if not driver.startswith('singlestore'):
                driver = f'singlestore+{driver}'
        else:
            driver = 'singlestore'
        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=driver,
        )

        self.database_name = alchemy_url.database
        super().do_connect(sqlalchemy.create_engine(alchemy_url))

    @property
    def funcs(self) -> FuncDict:
        """Return function dictionary."""
        if not hasattr(self, '_funcs'):
            self._funcs = FuncDict(self)
        return self._funcs

    @contextlib.contextmanager
    def begin(self) -> Iterator[Any]:
        """
        Begin a transaction context.

        Returns
        -------
        Iterator[Any]

        """
        with super().begin() as bind:
            previous_timezone = bind.execute(
                'SELECT @@session.time_zone',
            ).scalar()
            try:
                bind.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:
                warnings.warn(f"Couldn't set singlestore timezone: {str(e)}")

            try:
                yield bind
            finally:
                query = "SET @@session.time_zone = '{}'"
                bind.execute(query.format(previous_timezone))

    def table(
        self,
        name: str,
        database: Optional[str] = None,
        schema: Optional[sch.Schema] = None,
    ) -> ir.TableExpr:
        """
        Return a database table object.

        Parameters
        ----------
        name : str
            Name of the table to retrieve
        database : str, optional
            Database in which the table referred to by `name` resides. If
            ``None`` then the ``current_database`` is used.
        schema : str, optional
            Schema in which the table resides.  If ``None`` then the
            `public` schema is assumed.

        Returns
        -------
        ir.TableExpr

        """
        if database is not None and database != self.current_database:
            return self.database(name=database).table(name=name, schema=schema)
        else:
            alch_table = self._get_sqla_table(name, schema=schema)
            node = self.table_class(alch_table, self, self._schemas.get(name))
            return self.table_expr_class(node)

    def create_external_function(
        self,
        name: str,
        args: Tuple[FuncParam, ...],
        returns: Union[str, FuncReturns],
        remote_service: str,
        format: str = 'json',
        link: Optional[str] = None,
        if_exists: str = 'error',
        database: Optional[str] = None,
    ) -> Union[SingleStoreUDF, SingleStoreUDA]:
        """
        Create an external function.

        Parameters
        ----------
        name : str
            Name of the function
        args : Tuple[FuncArg]
            Tuple of function arguments
        returns : str or FuncReturns
            Output data type
        remote_service : str
            URL of the remote service
        format : str, optional
            Output data format: 'json' or 'rowdat_1'
        link : str, optional
            Link that stores connection details
        if_exists : str, optional
            Action to perform if the function already exists
        database : str, optional
            Name of database to insert the function into

        Returns
        -------
        SingleStoreUDF or SingleStoreUDA

        """
        orig_name = name

        # TODO: escape literals
        sql = ['CREATE']
        if if_exists == 'replace':
            sql.append('OR REPLACE')
        sql.append('EXTERNAL FUNCTION')

        if database:
            name = f'{database}.{name}'
            link = link and f'{database}.{link}' or None

        sql.append(name)

        sql.append('(')
        if args:
            sig = []
            for item in args:
                arg = f'{item.name} {item.dtype}'
                if item.default:
                    arg += ' DEFAULT {item.default}'
                arg += item.is_nullable and ' NULL' or ' NOT NULL'
                if item.collate:
                    arg += f' {item.collate}'
                sig.append(arg)
            sql.append(', '.join(sig))
        sql.append(')')

        if isinstance(returns, str):
            returns = FuncReturns(returns)

        sql.append(f'RETURNS {returns.dtype}')
        sql.append(returns.is_nullable and 'NULL' or 'NOT NULL')
        sql.append(f'AS REMOTE SERVICE "{remote_service}"')
        sql.append(f'FORMAT {format}')

        if link:
            sql.append(f'LINK {link}')

        self.raw_sql(' '.join(sql))

        self.funcs(refresh=True)

        return getattr(self.funcs, orig_name)

    def create_function(
        self,
        func: Callable[..., Any],
        database: Optional[str] = None,
    ) -> Union[SingleStoreUDF, SingleStoreUDA]:
        """
        Create a function within SingleStore from Python source.

        Parameters
        ----------
        func : Function
            Python function
        database : string, optional
            Name of the database to upload to. The current database
            is used by default.

        Returns
        -------
        SingleStoreUDF | SingleStoreUDA

        """
        import inspect
        import os
        import tempfile

        database = database or self.current_database

        TYPE_MAP = {
            int: 'int64',
            float: 'double',
            str: 'varchar(255)',
        }

        argspec = inspect.getfullargspec(func)
        anno = argspec.annotations
        inputs = []
        output = TYPE_MAP.get(anno.get('return', ''), '')

        for arg in argspec.args:
            inputs.append(TYPE_MAP[anno[arg]])

        with tempfile.TemporaryDirectory() as tmp:
            tmpf = os.path.join(tmp, 'func.py')
            with open(tmpf, 'w') as outfile:
                outfile.write(inspect.getsource(func))

            # Create function object
            out = wrap_udf(tmpf, inputs, output, func.__name__)

            # TODO: Support UDAs too.
            self.raw_sql(ddl.CreateUDF(out, name=out.name, database=database).compile())

            # Register the function with Ibis
            out.register(out.name, database)

            return out


@dt.dtype.register((singlestore.DOUBLE, singlestore.REAL))
def singlestore_double(satype: Any, nullable: bool = True) -> dt.DataType:
    """
    Register a double data type.

    Parameters
    ----------
    satype : Any
        SQLAlchemy data type
    nullable : bool, optional
        Is the column nullable?

    Returns
    -------
    dt.DataType

    """
    return dt.Double(nullable=nullable)


@dt.dtype.register(singlestore.FLOAT)
def singlestore_float(satype: Any, nullable: bool = True) -> dt.DataType:
    """
    Register a float data type.

    Parameters
    ----------
    satype : Any
        SQLAlchemy data type
    nullable : bool, optional
        Is the column nullable?

    Returns
    -------
    dt.DataType

    """
    return dt.Float(nullable=nullable)


@dt.dtype.register(singlestore.TINYINT)
def singlestore_tinyint(satype: Any, nullable: bool = True) -> dt.DataType:
    """
    Register a tiny int data type.

    Parameters
    ----------
    satype : Any
        SQLAlchemy data type
    nullable : bool, optional
        Is the column nullable?

    Returns
    -------
    dt.DataType

    """
    return dt.Int8(nullable=nullable)


@ dt.dtype.register(singlestore.BLOB)
def singlestore_blob(satype: Any, nullable: bool = True) -> dt.DataType:
    """
    Register a blob data type.

    Parameters
    ----------
    satype : Any
        SQLAlchemy data type
    nullable : bool, optional
        Is the column nullable?

    Returns
    -------
    dt.DataType

    """
    return dt.Binary(nullable=nullable)
