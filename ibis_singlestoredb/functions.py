#!/usr/bin/env python3
from __future__ import annotations

import re
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


def quote_identifier(value: str) -> str:
    return f'`{value}`'


_db2py_type: Dict[str, str] = {
    'int64': 'int',
    'bigint': 'int',
    'string': 'str',
    'varchar': 'str',
    'text': 'str',
}


def _get_py_type(typ: Optional[str]) -> str:
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
    return _db2py_type.get(typ, typ)


def build_function(conn: Any, name: str) -> Callable[..., Any]:
    db = quote_identifier(conn.database_name)
    qname = quote_identifier(name)
    proto = conn.raw_sql(f'show create function {db}.{qname}').fetchall()[0][2]
    proto = re.split(r'\bfunction\s+', proto, flags=re.I)[-1]
    name, proto = proto.split('(', 1)

    if re.search(r'\)\s+returns\s+', proto, flags=re.I):
        sig, ret = re.split(r'\)\s+returns\s+', proto, flags=re.I)
        ret, ftype = re.split(r'\s+as\s+', ret, flags=re.I)
    else:
        ret = None
        sig, ftype = re.split(r'\s+as\s+', proto, flags=re.I)

    ftype, info = ftype.split(' ', 1)
    ftype = ftype.strip()

    m = re.search(r"^(.*)'\s+format\s+(\w+)\s*;\s*$", info, flags=re.I)
    if m is None:
        code = ''
        format = ''
    else:
        code = m.group(1)
        format = m.group(2)

    if name.startswith('`'):
        name = name[1:-1]

    input_names: List[str] = []
    inputs: List[str] = []
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

    inputs = [
        dict(
            bigint='int64', text='string', varchar='string',
            double='double',
        )[x] for x in inputs
    ]

    out_nullable = False
    output = ret
    if output:
        out_nullable = not re.search(r'\bnot\s+null\b', output, flags=re.I)
        m = re.match(r'^\s*(\w+)', output)
        if m is None:
            raise ValueError(f'Could not extract nullable information from: {output}')
        output = dict(
            bigint='int64', text='string',
            varchar='string', double='double',
        )[m.group(1)]

    return _make_udf(
        name, ftype.lower(),
        list(zip(input_names, inputs, nullable)),
        (output, out_nullable), code, format,
    )


def _make_func_doc(
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
        dtype = _get_py_type(dtype)
        arg = f'{name} : {dtype}'
        if nullable:
            arg += ' or None'
        doc.append(arg)
    if output and output[0]:
        doc.append('')
        doc.extend(['Returns', '-------'])
        ret = '{}'.format(_get_py_type(output[0]))
        if output[1]:
            ret += ' or None'
        doc.append(ret)
    doc.append('')
    return '\n'.join(doc)


def _make_udf(
    name: str,
    ftype: str,
    inputs: Sequence[Tuple[str, str, bool]],
    output: Optional[Tuple[str, bool]],
    code: str,
    format: str,
) -> Callable[..., Any]:
    """Define a callable that invokes a UDF on the server."""
    # Define generic function API
    cls_params = {}
    for arg, dtype, nullable in inputs:
        cls_params[arg] = getattr(rlz, dtype)
    if output:
        cls_params['output_dtype'] = getattr(dt, output[0])
        cls_params['output_shape'] = rlz.shape_like(inputs[0][0])
    func_type = type(name.title(), (ops.ValueOp,), cls_params)

    def eval_func(*args: Any) -> types.Expr:
        return func_type(*args).to_expr()

    eval_func.__name__ = name
    eval_func.__qualname__ = f'ibis.singlestoredb.{name}'
    eval_func.__doc__ = _make_func_doc(name, ftype, inputs, output, code, format)

    # TODO: Check for existing function
    if output:
        if output[0] == 'string':
            setattr(types.StringValue, name, eval_func)
        elif output[0] == 'int':
            setattr(types.IntegerValue, name, eval_func)

    @ibis.singlestoredb.add_operation(func_type)
    def _eval_func(t: tr.ExprTranslator, expr: types.Expr) -> types.Expr:
        return getattr(sa.func, name)(*[t.translate(x) for x in expr.op().args])

    return eval_func
