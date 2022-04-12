# coding: utf-8
# Created on 12/04/2022 15:32
# Author : matteo

# ====================================================
# imports
from .utils import get_TDF


# ====================================================
# code
def test_at_indexer():
    # TDF
    TDF = get_TDF('1')

    # get value
    assert TDF.at[10, 'col1'] == 10

    # set value
    TDF.at[10, 'col1'] = -1
    assert TDF.values_num[60, 0] == -1

    # view
    view = TDF['0h', 50:70, ['col3', 'col1']]

    # get value
    assert view.at[50, 'col3'] == '250'

    # set value
    view.at[50, 'col3'] = '-5'
    assert TDF.values_str[0, 0] == '-5'


def test_iat_indexer():
    # TDF
    TDF = get_TDF('2')

    # get value
    assert TDF.iat[60, 0] == 10

    # set value
    TDF.iat[60, 0] = -1
    assert TDF.values_num[60, 0] == -1

    # view
    view = TDF['0h', 50:70, ['col3', 'col1']]

    # get value
    assert view.iat[0, 1] == '250'

    # set value
    view.iat[0, 1] = '-5'
    assert TDF.values_str[0, 0] == '-5'


def test_loc_indexer():
    # TDF
    # get value
    ...

    # set value

    # view
    # get value

    # set value


def test_iloc_indexer():
    # TDF
    # get value
    ...

    # set value

    # view
    # get value

    # set value


if __name__ == '__main__':
    test_at_indexer()
    test_iat_indexer()
    test_loc_indexer()
    test_iloc_indexer()
