# coding: utf-8
# Created on 18/01/2021 11:18
# Author : matteo

# ====================================================
# imports

# ====================================================
# code


TemporalDataFrame_internal_attributes = ['_name', 'name', '_time_points', 'time_points', '_time_points_column_name',
                                         '_time_points_column', 'time_points_column', 'time_points_column_name',
                                         'n_time_points', '_index', 'index', 'n_index_total', '_columns', 'columns',
                                         'n_columns', '_df', 'shape', 'dtypes', 'values', 'axes', 'ndim', 'size',
                                         'empty', 'at', 'iat', 'loc', 'iloc', '_is_backed', '_is_locked', '_file',
                                         'file']

ViewTemporalDataFrame_internal_attributes = TemporalDataFrame_internal_attributes + \
                                            ['parent', '_tp_slicer', '_parent_data', '_index', '_columns', '_lock']

TemporalDataFrame_reserved_keys = ['Time_Point']
