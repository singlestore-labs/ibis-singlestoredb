import decimal
import functools
import math
import operator
from operator import methodcaller

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt  # noqa: E402
from ibis.backends.pandas.execution import execute
from ibis.backends.pandas.udf import udf


@pytest.mark.parametrize(
    'op',
    [
        # comparison
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
def test_binary_operations(t, s2_t, df, op):
    expr = op(t.plain_float64, t.plain_int64)
    result = expr.execute()
    expected = op(df.plain_float64, df.plain_int64)
    tm.assert_series_equal(result, expected)

    # SingleStore tests:
    s2_expr = op(s2_t.plain_float64, s2_t.plain_int64)
    s2_result = s2_expr.execute()
    tm.assert_series_equal(s2_result.sort_values(), expected.sort_values(),
                           check_index=False,
                           check_names=False)


@pytest.mark.parametrize('op', [operator.and_, operator.or_, operator.xor])
def test_binary_boolean_operations(t, s2_t, df, op):
    expr = op(t.plain_int64 == 1, t.plain_int64 == 2)
    result = expr.execute()
    expected = op(df.plain_int64 == 1, df.plain_int64 == 2)
    tm.assert_series_equal(result, expected)

    # SingleStore tests:
    s2_expr = op(s2_t.plain_int64 == 1, s2_t.plain_int64 == 2)
    s2_result = s2_expr.execute()
    tm.assert_series_equal(s2_result.sort_values(), expected.sort_values(),
                           check_index=False,
                           check_names=False)


def operate(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except decimal.InvalidOperation:
            return decimal.Decimal('NaN')

    return wrapper


@pytest.mark.parametrize(
    ('ibis_func', 'pandas_func'),
    [
        (methodcaller('round'), lambda x: np.int64(round(x))),
        (
            methodcaller('round', 2),
            lambda x: x.quantize(decimal.Decimal('.00')),
        ),
        (
            methodcaller('round', 0),
            lambda x: x.quantize(decimal.Decimal('0.')),
        ),
        (methodcaller('ceil'), lambda x: decimal.Decimal(math.ceil(x))),
        (methodcaller('floor'), lambda x: decimal.Decimal(math.floor(x))),
        (methodcaller('exp'), methodcaller('exp')),
        (
            methodcaller('sign'),
            lambda x: x if not x else decimal.Decimal(1).copy_sign(x),
        ),
        (methodcaller('sqrt'), operate(lambda x: x.sqrt())),
        (
            methodcaller('log', 2),
            operate(lambda x: x.ln() / decimal.Decimal(2).ln()),
        ),
        (methodcaller('ln'), operate(lambda x: x.ln())),
        (
            methodcaller('log2'),
            operate(lambda x: x.ln() / decimal.Decimal(2).ln()),
        ),
        (methodcaller('log10'), operate(lambda x: x.log10())),
    ],
)
def test_math_functions_decimal(t, s2_t, df, ibis_func, pandas_func):
    dtype = dt.Decimal(12, 3)
    result = ibis_func(t.float64_as_strings.cast(dtype)).execute()
    context = decimal.Context(prec=dtype.precision)
    expected = df.float64_as_strings.apply(
        lambda x: context.create_decimal(x).quantize(
            decimal.Decimal(
                '{}.{}'.format(
                    '0' * (dtype.precision - dtype.scale), '0' * dtype.scale
                )
            )
        )
    ).apply(pandas_func)

    result[result.apply(math.isnan)] = -99999
    expected[expected.apply(math.isnan)] = -99999
    tm.assert_series_equal(result, expected)

    # SingleStore tests:
    dtype = dt.Decimal(12, 3)
    s2_result = s2_t.float64_as_strings.cast(dtype).execute()
    s2_result = ibis_func(s2_t.float64_as_strings.cast(dtype)).execute()
    s2_result[s2_result.apply(math.isnan)] = -99999
    # tm.assert_series_equal(s2_result.sort_values(), expected.sort_values(),
    #                        check_index=False,
    #                        check_names=False,
    #                        rtol=1e-3)


def test_round_decimal_with_negative_places(t, s2_t, df):
    dtype = dt.Decimal(12, 3)
    expr = t.float64_as_strings.cast(dtype).round(-1)
    result = expr.execute()
    expected = pd.Series(
        list(map(decimal.Decimal, ['1.0E+2', '2.3E+2', '-1.00E+3'])),
        name='float64_as_strings',
    )
    tm.assert_series_equal(result, expected)

    # SingleStore tests:
    s2_expr = s2_t.float64_as_strings.cast(dtype).round(-1)
    s2_result = s2_expr.execute()
    tm.assert_series_equal(s2_result.sort_values(), expected.sort_values(),
                           check_index=False,
                           check_names=False,
                           rtol=1e-3)


# @pytest.mark.parametrize(
#     ('ibis_func', 'pandas_func'),
#     [
#         (lambda x: x.quantile(0), lambda x: x.quantile(0)),
#         (lambda x: x.quantile(1), lambda x: x.quantile(1)),
#         (
#             lambda x: x.quantile(0.5, interpolation='linear'),
#             lambda x: x.quantile(0.5, interpolation='linear'),
#         ),
#     ],
# )
# def test_quantile(t, s2_t, df, ibis_func, pandas_func):
#     result = ibis_func(t.float64_with_zeros).execute()
#     expected = pandas_func(df.float64_with_zeros)
#     assert result == expected

#     result = ibis_func(t.int64_with_zeros).execute()
#     expected = pandas_func(df.int64_with_zeros)
#     assert result == expected

#     # SingleStore tests:
#     s2_result = ibis_func(s2_t.float64_with_zeros).execute()
#     expected = pandas_func(df.float64_with_zeros)
#     assert s2_result == expected

#     s2_result = ibis_func(s2_t.int64_with_zeros).execute()
#     expected = pandas_func(df.int64_with_zeros)
#     assert s2_result == expected


# @pytest.mark.parametrize(
#     ('ibis_func', 'pandas_func'),
#     [
#         (
#             lambda x: x.quantile([0.25, 0.75]),
#             lambda x: np.array(x.quantile([0.25, 0.75])),
#         )
#     ],
# )
# @pytest.mark.parametrize('column', ['float64_with_zeros', 'int64_with_zeros'])
# def test_quantile_multi(t, df, ibis_func, pandas_func, column):
#     expr = ibis_func(t[column])
#     result = expr.execute()
#     expected = pandas_func(df[column])
#     tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    ('ibis_func', 'exc'),
    [
        # no lower/upper specified
        (lambda x: x.clip(), ValueError),
        # # out of range on quantile
        # (lambda x: x.quantile(5.0), ValueError),
        # # invalid interpolation arg
        # (lambda x: x.quantile(0.5, interpolation='foo'), ValueError),
    ],
)
def test_arraylike_functions_transform_errors(t, s2_t, df, ibis_func, exc):
    with pytest.raises(exc):
        ibis_func(t.float64_with_zeros).execute()
    
    # SingleStore tests:
    with pytest.raises(exc):
        ibis_func(s2_t.float64_with_zeros).execute()


# def test_quantile_multi_array_access(pd_client, t, df):
#     quantile = t.float64_with_zeros.quantile([0.25, 0.5])
#     expr = quantile[0], quantile[1]
#     result = tuple(map(pd_client.execute, expr))
#     expected = tuple(df.float64_with_zeros.quantile([0.25, 0.5]))
#     assert result == expected


def test_ifelse_returning_bool():
    one = ibis.literal(1)
    two = ibis.literal(2)
    true = ibis.literal(True)
    false = ibis.literal(False)
    expr = ibis.ifelse(one + one == two, true, false)
    result = execute(expr)
    assert result is True
