# coding: utf-8
# Created on 29/01/2021 10:13
# Author : matteo

# ====================================================
# imports
import vdata


# ====================================================
# code
def test_TDF_sub_setting():
    data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

    TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                  time_col=None, time_points=['0h', '5h', '10h'],
                                  index=['a', 'b', 'c'], name=1)
    print(TDF['0h'])


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    test_TDF_sub_setting()
