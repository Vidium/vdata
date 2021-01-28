# coding: utf-8
# Created on 18/01/2021 11:18
# Author : matteo

# ====================================================
# imports

# ====================================================
# code
TemporalDataFrame_internal_attributes = ['_time_points_col', '_df', '_time_points', 'TP_from_DF', 'name', '_name',
                                         'time_points', 'time_points_column_name', 'time_points_column', 'columns',
                                         'index', 'n_time_points', 'n_columns', 'dtypes', 'values', 'axes', 'ndim',
                                         'size', 'shape', 'empty', 'at', 'iat', 'loc', 'iloc', '_len_index', '_columns']

TemporalDataFrame_reserved_keys = ['__TPID', 'df_data']