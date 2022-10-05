#!/usr/bin/env python3
from __future__ import annotations

import re
import warnings
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import ibis
import ibis.backends.base.sql.compiler.translator as tr
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as types
import sqlalchemy as sa


class Table(dt.DataType):
    """Placeholder for Table types until Ibis supports them."""


def _build_data_type(
    dtype: str,
    info: Dict[str, Any],
    args: Optional[Sequence[int]] = None,
    schema: Optional[Sequence[dt.DataType]] = None,
) -> dt.DataType:
    """
    Build a dt.DataType from given SingleStoreDB data information.

    Parameters
    ----------
    dtype : str
        Name of the data type
    info : Dict[str, Any]
        Data type modifiers
    args : Sequence[int], optional
        Data type parameters
    schema : Sequence[dt.DataType], optional
        Schema of data type if it is a array, record, or table

    Returns
    -------
    dt.DataType

    """
    type_map = {
        'bool': dt.Boolean, 'bit': dt.Binary, 'tinyint': dt.Int8,
        'smallint': dt.Int16, 'mediumint': dt.Int32, 'int': dt.Int32,
        'bigint': dt.Int64, 'float': dt.Float32, 'double': dt.Float64,
        'tinyint unsigned': dt.UInt8, 'smallint unsigned': dt.UInt16,
        'mediumint unsigned': dt.UInt32, 'int unsigned': dt.UInt32,
        'bignt unsigned': dt.UInt64,
        'decimal': dt.Decimal, 'date': dt.Date, 'time': dt.Interval,
        'datetime': dt.Timestamp, 'timestamp': dt.Timestamp, 'year': dt.Int16,
        'char': dt.String, 'varchar': dt.String, 'text': dt.String,
        'tinytext': dt.String, 'mediumtext': dt.String, 'longtext': dt.String,
        'binary': dt.Binary, 'varbinary': dt.Binary, 'blob': dt.Binary,
        'tinyblob': dt.Binary, 'mediumblob': dt.Binary, 'longblob': dt.Binary,
        'json': dt.JSON, 'record': dt.Struct, 'geograph': dt.Geography,
        'geographypoint': dt.Point, 'table': Table, 'null': dt.Null,
        'array': dt.Array,
    }

    attrs: Dict[str, Any] = {}

    if args and 'decimal' in dtype:
        attrs['precision'] = args[0]
        if len(args) > 1:
            attrs['scale'] = args[1]

    if dtype in ['datetime', 'timestamp']:
        attrs['value_type'] = dt.Int64()

    elif dtype in ['record']:
        if not schema:
            raise ValueError('A schema is required for record and table types.')
        attrs['names'] = [x[0] for x in schema]
        attrs['types'] = [x[1] for x in schema]

    elif dtype in ['array']:
        if not schema:
            raise ValueError('A data type is required for array types.')
        attrs['value_type'] = schema[0][1]

    attrs['nullable'] = info.get('nullable', False)

    return type_map[dtype](**attrs)


def _parse_data_type(data: str) -> Tuple[dt.DataType, str]:
    """
    Parse data type from string.

    Parameters
    ----------
    data : str
        String to parse

    Returns
    -------
    Tuple of parsed data type and remaining string

    """
    _, data_type, data = re.split(
        r'\s*(\w+(?:\s+UNSIGNED)?)\s*', data, flags=re.I, maxsplit=1,
    )
    data_type = data_type.lower()

    schema = None
    data_type_args: List[int] = []

    if data_type in ['table', 'record', 'array']:
        _, data = re.split(r'^\s*\(\s*', data, flags=re.I, maxsplit=1)
        if data_type in ['array']:
            schema, data = _parse_params(data, parse_names=False)
        else:
            schema, data = _parse_params(data)
    elif data.startswith('('):
        _, data = re.split(r'^\s*\(\s*', data, maxsplit=1)
        args, data = re.split(r'\s*\)\s*', data, flags=re.I, maxsplit=1)
        data_type_args = [int(x) for x in re.split(r'\s*,\s*', args)]

    modifiers = {}
    while data and re.match(
        r'^(CHARACTER\s+SET\s+|COLLATE\s+|NULL|NOT\s+NULL)',
        data, flags=re.I,
    ):
        if re.match(r'CHARACTER\s+SET\s+', data, flags=re.I):
            _, modifiers['character_set'], data = re.split(
                r'CHARACTER\s+SET\s+(\S+)\s*',
                data, flags=re.I, maxsplit=1,
            )
        elif re.match(r'COLLATE\s+', data, flags=re.I):
            _, modifiers['collate'], data = re.split(
                r'COLLATE\s+(\S+)\s*',
                data, flags=re.I, maxsplit=1,
            )
        elif re.match(r'NULL', data, flags=re.I):
            _, data = re.split(r'NULL\s*', data, flags=re.I, maxsplit=1)
            modifiers['nullable'] = True
        elif re.match(r'NOT\s+NULL', data, flags=re.I):
            _, data = re.split(r'NOT\s+NULL\s*', data, flags=re.I, maxsplit=1)
            modifiers['nullable'] = False
        else:
            unknown, data = re.split(r'(\S+)', data, flags=re.I, maxsplit=1)
            warnings.warn(
                f'Skipping unknown CREATE FUNCTION syntax: {unknown}',
                RuntimeWarning,
            )

    return _build_data_type(data_type, modifiers, data_type_args, schema), data


def _parse_params(params: str, parse_names: bool = True) -> Tuple[List[dt.DataType], str]:
    """
    Parse function / table / record / array parameters from string.

    Parameters
    ----------
    params : str
        String to parse
    parse_names : bool, optional
        Do the parameters include parameter names?

    Returns
    -------
    List of parsed datatypes and remaining string

    """
    out = []
    i = ord('a')
    while params and not params.startswith(')'):
        if parse_names:
            _, param_name, params = re.split(r'(\S+)\s+', params, flags=re.I, maxsplit=1)

            if param_name.startswith('`') and param_name.endswith('`'):
                param_name = param_name[1:-1]
        else:
            param_name = chr(i)
            i += 1
            params = params.lstrip()

        param_type, params = _parse_data_type(params)

        if params.startswith(','):
            params = re.sub(r'^,\s*', r'', params)

        out.append((param_name, param_type))

    return out, re.sub(r'^\s*\)\s*', r'', params, flags=re.I)


def _parse_create_function(
    func: str,
) -> Tuple[str, str, List[dt.DataType], dt.DataType, Dict[str, Any]]:
    """
    Parse a CREATE FUNCTION prototype.

    Parameters
    ----------
    func : str
        String to parse

    Returns
    -------
    Tuple of function type, function name, list of data types, return data type,
    and dictionary of function metadata

    """
    # Strip CREATE keyword
    func = re.sub(r'^\s*CREATE\s+(OR\s+REPLACE\s+)?\s*', r'', func, flags=re.I)

    # Get function type
    _, func_type, func = re.split(
        r'((?:EXTERNAL\s+)?(?:FUNCTION|AGGREGATE))\s+',
        func, flags=re.I, maxsplit=1,
    )
    func_type = func_type.lower()

    # Get function name
    func_name, func = re.split(r'\s*\(\s*', func, flags=re.I, maxsplit=1)
    if func_name.startswith('`') and func_name.endswith('`'):
        func_name = func_name[1:-1]

    # Parse parameters
    inputs, func = _parse_params(func, parse_names='agg' not in func_type)

    # Bypass RETURNS keyword
    func = re.sub(r'^\s*RETURNS\s+', r'', func, flags=re.I)

    # Parse return type
    return_type, func = _parse_data_type(func)

    # Detect implementation-specific information
    info: Dict[str, Any] = {}

    m_wasm = re.search(r'\bAS\s+WASM\b', func, flags=re.I)
    if m_wasm:
        info['wasm'] = True

    m_remote_service = re.search(
        r'\bAS\s+REMOTE\s+SERVICE\s*("|\')([^\1])\1', func, flags=re.I,
    )
    if m_remote_service:
        info['remote_service'] = m_remote_service.group(2)

    m_format = re.search(r'\bFORMAT\s+(\S+)', func, flags=re.I)
    if m_format:
        info['format'] = m_format.group(1).lower()

    return func_type, func_name, inputs, return_type, info


def build_function(conn: Any, name: str) -> Optional[Callable[..., Any]]:
    """
    Build a Ibis function from a given function name.

    Parameters
    ----------
    conn : Ibis Connection
        Ibis connection object
    name : str
        Name of the function to build

    Returns
    -------
    Callable Ibis function

    """
    quote_identifier = conn.con.dialect.identifier_preparer.quote_identifier
    db = quote_identifier(conn.database_name)
    qname = quote_identifier(name)
    proto = conn.raw_sql(f'show create function {db}.{qname}').fetchall()[0][2]

    try:
        func_type, func_name, inputs, output, info = _parse_create_function(proto)
    except Exception:
        print(f'Failed to parse: {proto}')
        return None

    return _make_udf(func_name, func_type, inputs, output, info)


def _make_func_doc(
    name: str,
    ftype: str,
    inputs: Sequence[Tuple[str, dt.DataType]],
    output: Optional[dt.DataType],
    info: Optional[Dict[str, Any]],
) -> str:
    """
    Construct the docstring using the function information.

    Parameters
    ----------
    name : str
        Name of the function
    ftype : str
        Type of the function in the database
    inputs : Sequence[Tuple[str, dt.DataType]]
        Sequence of (name, type) elements describing
        the inputs of the function
    output : dt.DataType, optional
        Data type of the return value of the function
    info : Dict[str, Any], optional
        Function metadata

    Returns
    -------
    str

    """
    info = info or {}
    code = info.get('code', '')
    format = info.get('format', '')
    doc = [f'Call `{name}` {ftype} function.', '']
    if ftype == 'remote service':
        doc.append(f'Accesses remote service at {code} using {format} format.')
        doc.append('')
    doc.extend(['Parameters', '----------'])
    for name, dtype in inputs:
        arg = f'{name} : {dtype}'
        if dtype.nullable:
            arg += ' or None'
        doc.append(arg)
    if output is not None:
        doc.append('')
        doc.extend(['Returns', '-------'])
        ret = str(output)
        if output.nullable:
            ret += ' or None'
        doc.append(ret)
    doc.append('')
    return '\n'.join(doc)


def _make_udf(
    name: str,
    ftype: str,
    inputs: Sequence[Tuple[str, dt.DataType]],
    output: Optional[dt.DataType],
    info: Optional[Dict[str, Any]],
) -> Optional[Callable[..., Any]]:
    """Define a callable that invokes a UDF on the server."""
    if output is not None and isinstance(output, Table):
        warnings.warn(
            f'Could not create function `{name}`. '
            'Table return types are not supported.',
            RuntimeWarning,
        )
        return None

    info = info or {}

    # Define generic function API
    cls_params = {}

    for arg, dtype in inputs:
        cls_params[arg] = rlz.value(dtype)

    if output:
        cls_params['output_dtype'] = output
        # TODO: Verify shape for various output types
        if inputs:
            cls_params['output_shape'] = rlz.shape_like(inputs[0][0])

    func_type = type(name.title().replace('_', ''), (ops.ValueOp,), cls_params)

    def eval_func(*args: Any) -> types.Expr:
        return func_type(*args).to_expr()

    eval_func.__name__ = name
    eval_func.__qualname__ = f'ibis.singlestoredb.{name}'
    eval_func.__doc__ = _make_func_doc(name, ftype, inputs, output, info)

    # TODO: Check for existing function
    if inputs:
        if isinstance(inputs[0][1], dt.String):
            setattr(types.StringValue, name, eval_func)
        elif isinstance(inputs[0][1], dt.Integer):
            setattr(types.IntegerValue, name, eval_func)
        elif isinstance(inputs[0][1], dt.Floating):
            setattr(types.FloatingValue, name, eval_func)

    @ibis.singlestoredb.add_operation(func_type)
    def _eval_func(t: tr.ExprTranslator, expr: types.Expr) -> types.Expr:
        return getattr(sa.func, name)(*[t.translate(x) for x in expr.op().args])

    return eval_func
