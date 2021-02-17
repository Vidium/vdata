# coding: utf-8
# Created on 29/01/2021 10:13
# Author : matteo

# ====================================================
# imports
import vdata


# ====================================================
# code
def test_TDF_sub_setting():
    data = {'col1': [1., 2., 3., 4., 5., 6., 7., 8., 9.]}

    TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                  time_col_name=None, time_points=['0h', '5h', '10h'],
                                  index=['a', 'b', 'c'], name=1)
    assert repr(TDF['0h']) == "View of TemporalDataFrame '1'\n" \
                              "\033[4mTime point : 0.0 hours\033[0m\n" \
                              "   col1\n" \
                              "a   1.0\n" \
                              "b   2.0\n" \
                              "c   3.0\n\n", repr(TDF['0h'])

    try:
        print(TDF['15h'])

    except vdata.VValueError as e:
        assert e.msg == "Time points not found in this TemporalDataFrame."

    assert repr(TDF['0h', 'z']) == "Empty View of TemporalDataFrame '1'\n" \
                                   "Time points: []\n" \
                                   "Columns: ['col1']\n" \
                                   "Index: []", repr(TDF['0h', 'z'])

    assert TDF['0h'].time_points == [vdata.TimePoint('0h')]

    TDF = vdata.TemporalDataFrame(data=data, time_list=[0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.],
                                  time_col_name=None, time_points=[0., .5, 1.],
                                  index=['a', 'b', 'c'], name=1)
    assert repr(TDF[0]) == "View of TemporalDataFrame '1'\n" \
                           "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                           "   col1\n" \
                           "a   1.0\n" \
                           "b   2.0\n" \
                           "c   3.0\n\n", repr(TDF[0])


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    test_TDF_sub_setting()
