import decimal

import numpy as np
import pandas as pd
import pytest

import ibis.expr.datatypes as dt
from ibis.backends.pandas import Backend

import ibis

import os
from pathlib import Path

USER = os.environ.get('IBIS_TEST_SINGLESTORE_USER', 'ibis')
PASSWORD = os.environ.get('IBIS_TEST_SINGLESTORE_PASSWORD', 'ibis')
HOST = os.environ.get('IBIS_TEST_SINGLESTORE_HOST', 'localhost')
PORT = os.environ.get('IBIS_TEST_SINGLESTORE_PORT', 3306)
DATABASE = os.environ.get('IBIS_TEST_SINGLESTORE_DATABASE', 'ibis_testing')

@pytest.fixture(scope='session')
def data_directory() -> Path:
    """Return the test data directory.
    Returns
    -------
    Path
        Test data directory
    """
    root = Path(__file__).absolute().parents[2]

    return Path(
        os.environ.get(
            "IBIS_TEST_DATA_DIRECTORY",
            root / "ci" / "ibis-testing-data",
        )
    )

@pytest.fixture(scope='module')
def df():
    return pd.DataFrame(
        {
            'plain_int64': list(range(1, 4)),
            'plain_strings': list('abc'),
            'plain_float64': [4.0, 5.0, 6.0],
            'plain_datetimes_naive': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=3
                ).values
            ),
            'plain_datetimes_ny': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=3
                ).values
            ).dt.tz_localize('UTC'),
            'plain_datetimes_utc': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=3
                ).values
            ).dt.tz_localize('UTC'),
            'dup_strings': list('dad'),
            'dup_ints': [1, 2, 1],
            'float64_as_strings': ['100.01', '234.23', '-999.34'],
            'int64_as_strings': list(map(str, range(1, 4))),
            'strings_with_space': [' ', 'abab', 'ddeeffgg'],
            'int64_with_zeros': [0, 1, 0],
            'float64_with_zeros': [1.0, 0.0, 1.0],
            'float64_positive': [1.0, 2.0, 1.0],
            'strings_with_nulls': ['a', None, 'b'],
            'datetime_strings_naive': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=3
                ).values
            ).astype(str),
            'datetime_strings_ny': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=3
                ).values
            )
            .dt.tz_localize('America/New_York')
            .astype(str),
            'datetime_strings_utc': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=3
                ).values
            )
            .dt.tz_localize('UTC')
            .astype(str),
            'decimal': list(map(decimal.Decimal, [1.0, 2, 3.234])),
            # 'array_of_float64': [
            #     np.array([1.0, 2.0]),
            #     np.array([3.0]),
            #     np.array([]),
            # ],
            # 'array_of_int64': [np.array([1, 2]), np.array([]), np.array([3])],
            # 'array_of_strings': [
            #     np.array(['a', 'b']),
            #     np.array([]),
            #     np.array(['c']),
            # ],
            # 'map_of_strings_integers': [{'a': 1, 'b': 2}, None, {}],
            # 'map_of_integers_strings': [{}, None, {1: 'a', 2: 'b'}],
            # 'map_of_complex_values': [None, {'a': [1, 2, 3], 'b': []}, {}],
        }
    )


@pytest.fixture(scope='module')
def batting_df(data_directory):
    num_rows = 1000
    start_index = 30
    df = pd.read_csv(
        data_directory / 'batting.csv',
        index_col=None,
        sep=',',
        header=0,
        skiprows=range(1, start_index + 1),
        nrows=num_rows,
    )
    return df.reset_index(drop=True)


@pytest.fixture(scope='module')
def awards_players_df(data_directory):
    return pd.read_csv(
        data_directory / 'awards_players.csv',
        index_col=None,
        sep=',',
    )


@pytest.fixture(scope='module')
def df1():
    return pd.DataFrame(
        {'key': list('abcd'), 'value': [3, 4, 5, 6], 'key2': list('eeff')}
    )


@pytest.fixture(scope='module')
def df2():
    return pd.DataFrame(
        {'key': list('ac'), 'other_value': [4.0, 6.0], 'key3': list('fe')}
    )


@pytest.fixture(scope='module')
def intersect_df2():
    return pd.DataFrame(
        {'key': list('cd'), 'value': [5, 6], 'key2': list('ff')}
    )


@pytest.fixture(scope='module')
def time_df1():
    return pd.DataFrame(
        {'time': pd.to_datetime([1, 2, 3, 4]), 'value': [1.1, 2.2, 3.3, 4.4]}
    )


@pytest.fixture(scope='module')
def time_df2():
    return pd.DataFrame(
        {'time': pd.to_datetime([2, 4]), 'other_value': [1.2, 2.0]}
    )


@pytest.fixture(scope='module')
def time_df3():
    return pd.DataFrame(
        {
            'time': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=8
                ).values
            ),
            'id': list(range(1, 5)) * 2,
            'value': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
        }
    )


@pytest.fixture(scope='module')
def time_keyed_df1():
    return pd.DataFrame(
        {
            'time': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=6
                ).values
            ),
            'key': [1, 2, 3, 1, 2, 3],
            'value': [1.2, 1.4, 2.0, 4.0, 8.0, 16.0],
        }
    )


@pytest.fixture(scope='module')
def time_keyed_df2():
    return pd.DataFrame(
        {
            'time': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', freq='3D', periods=3
                ).values
            ),
            'key': [1, 2, 3],
            'other_value': [1.1, 1.2, 2.2],
        }
    )


@pytest.fixture(scope='module')
def pd_client(
    df,
    df1,
    df2,
    df3,
    time_df1,
    time_df2,
    time_df3,
    time_keyed_df1,
    time_keyed_df2,
    intersect_df2,
):
    return Backend().connect(
        {
            'df': df,
            'df1': df1,
            'df2': df2,
            'df3': df3,
            'left': df1,
            'right': df2,
            'time_df1': time_df1,
            'time_df2': time_df2,
            'time_df3': time_df3,
            'time_keyed_df1': time_keyed_df1,
            'time_keyed_df2': time_keyed_df2,
            'intersect_df2': intersect_df2,
        }
    )

s2_df_dict = dict({
    ('plain_int64', dt.int64),
    ('plain_strings', dt.string),
    ('plain_float64', dt.float64),
    ('plain_datetimes_naive', dt.timestamp),
    ('plain_datetimes_ny', dt.timestamp),
    ('plain_datetimes_utc', dt.timestamp),
    ('dup_strings', dt.string),
    ('dup_ints', dt.int64),
    ('float64_as_strings', dt.string),
    ('int64_as_strings', dt.string),
    ('strings_with_space', dt.string),
    ('int64_with_zeros', dt.int64),
    ('float64_with_zeros', dt.float64),
    ('float64_positive', dt.float64),
    ('strings_with_nulls', dt.string),
    ('datetime_strings_naive', dt.string),
    ('datetime_strings_ny', dt.string),
    ('datetime_strings_utc', dt.string),
    ('decimal', dt.Decimal(4, 3)),
})

s2_t_schema = ibis.Schema(
    s2_df_dict.keys(),
    s2_df_dict.values()
)

@pytest.fixture(scope='module')
def s2_client(
    df,
    df1,
    df2,
    df3,
    time_df1,
    time_df2,
    time_df3,
    time_keyed_df1,
    time_keyed_df2,
    intersect_df2,
):
    con = ibis.singlestore.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE,
        )
    con.create_table(name="df", expr=df, schema=s2_t_schema, force=True)
    con.create_table(name="df1", expr=df1, force=True)
    con.create_table(name="df2", expr=df2, force=True)
    con.create_table(name="df3", expr=df3, force=True)
    con.create_table(name="time_df1", expr=time_df1, force=True)
    con.create_table(name="time_df2", expr=time_df2, force=True)
    con.create_table(name="time_df3", expr=time_df3, force=True)
    con.create_table(name="time_keyed_df1", expr=time_keyed_df1, force=True)
    con.create_table(name="time_keyed_df2", expr=time_keyed_df2, force=True)
    con.create_table(name="intersect_df2", expr=intersect_df2, force=True)
    return con

@pytest.fixture(scope='module')
def df3():
    return pd.DataFrame(
        {
            'key': list('ac'),
            'other_value': [4.0, 6.0],
            'key2': list('ae'),
            'key3': list('fe'),
        }
    )


t_schema = {
    'decimal': dt.Decimal(4, 3),
    'array_of_float64': dt.Array(dt.double),
    'array_of_int64': dt.Array(dt.int64),
    'array_of_strings': dt.Array(dt.string),
    'map_of_strings_integers': dt.Map(dt.string, dt.int64),
    'map_of_integers_strings': dt.Map(dt.int64, dt.string),
    'map_of_complex_values': dt.Map(dt.string, dt.Array(dt.int64)),
}

@pytest.fixture(scope='module')
def t(pd_client):
    return pd_client.table('df', schema=t_schema)

@pytest.fixture(scope='module')
def s2_t(s2_client):
    return s2_client.table('df')

@pytest.fixture(scope='module')
def lahman(batting_df, awards_players_df):
    return Backend().connect(
        {'batting': batting_df, 'awards_players': awards_players_df}
    )

@pytest.fixture(scope='module')
def s2_lahman(batting_df, awards_players_df):
    con = ibis.singlestore.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database='lahman',
        )
    con.create_table(name="batting", expr=batting_df, force=True)
    con.create_table(name="awards_players", expr=awards_players_df, force=True)
    return con


@pytest.fixture(scope='module')
def left(pd_client):
    return pd_client.table('left')


@pytest.fixture(scope='module')
def right(pd_client):
    return pd_client.table('right')


@pytest.fixture(scope='module')
def time_left(pd_client):
    return pd_client.table('time_df1')


@pytest.fixture(scope='module')
def time_right(pd_client):
    return pd_client.table('time_df2')


@pytest.fixture(scope='module')
def time_table(pd_client):
    return pd_client.table('time_df3')


@pytest.fixture(scope='module')
def time_keyed_left(pd_client):
    return pd_client.table('time_keyed_df1')


@pytest.fixture(scope='module')
def time_keyed_right(pd_client):
    return pd_client.table('time_keyed_df2')


@pytest.fixture(scope='module')
def batting(lahman):
    return lahman.table('batting')

@pytest.fixture(scope='module')
def s2_batting(s2_lahman):
    return s2_lahman.table('batting')


@pytest.fixture(scope='module')
def sel_cols(batting):
    cols = batting.columns
    start, end = cols.index('AB'), cols.index('H') + 1
    return ['playerID', 'yearID', 'teamID', 'G'] + cols[start:end]


@pytest.fixture(scope='module')
def players_base(batting, sel_cols):
    return batting[sel_cols].sort_by(sel_cols[:3])


@pytest.fixture(scope='module')
def players(players_base):
    return players_base.groupby('playerID')


@pytest.fixture(scope='module')
def players_df(players_base):
    return players_base.execute().reset_index(drop=True)