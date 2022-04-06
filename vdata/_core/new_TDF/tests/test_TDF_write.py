# coding: utf-8
# Created on 30/03/2022 17:35
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np
from h5py import File, Dataset
from pathlib import Path

from typing import Union

from .utils import get_TDF
from ..dataframe import TemporalDataFrame
from ..view import ViewTemporalDataFrame


# ====================================================
# code
def check_H5_file(reference_TDF: Union[TemporalDataFrame, ViewTemporalDataFrame],
                  path: Path) -> None:
    """
    Check a TemporalDataFrame was correctly written to a H5 file at path <path>.
    """
    def get_dset(data: Dataset) -> np.ndarray:
        if data.dtype == np.dtype('O'):
            return data.asstr()[()]

        return data[()]

    with File(path, 'r') as h5_file:
        assert 'type' in h5_file.attrs.keys()
        assert h5_file.attrs['type'] == 'TDF'

        assert 'name' in h5_file.attrs.keys()
        assert h5_file.attrs['name'] == reference_TDF.name

        assert 'locked_indices' in h5_file.attrs.keys()
        assert h5_file.attrs['locked_indices'] == reference_TDF.has_locked_indices

        assert 'locked_columns' in h5_file.attrs.keys()
        assert h5_file.attrs['locked_columns'] == reference_TDF.has_locked_columns

        assert 'time_points_column_name' in h5_file.attrs.keys()
        assert h5_file.attrs['time_points_column_name'] == "__TDF_None__" if reference_TDF.timepoints_column_name is \
            None else reference_TDF.timepoints_column_name

        assert 'index' in h5_file.keys()
        assert np.all(get_dset(h5_file['index']) == reference_TDF.index)

        assert 'columns_numerical' in h5_file.keys()
        assert np.all(get_dset(h5_file['columns_numerical']) == reference_TDF.columns_num)

        assert 'columns_string' in h5_file.keys()
        assert np.all(get_dset(h5_file['columns_string']) == reference_TDF.columns_str)

        assert 'timepoints' in h5_file.keys()
        assert np.all(get_dset(h5_file['timepoints']) == reference_TDF.timepoints_column_str)

        assert 'values_numerical' in h5_file.keys()
        assert np.all(get_dset(h5_file['values_numerical']) == reference_TDF.values_num)

        assert 'values_string' in h5_file.keys()
        assert np.all(get_dset(h5_file['values_string']) == reference_TDF.values_str)


def cleanup(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink(missing_ok=True)


def test_write():
    output_file = Path(__file__).parent

    # create TDF
    TDF = get_TDF('1')

    # create exact copy as reference
    reference_TDF = get_TDF('1')

    # create TDF with different values
    modified_TDF = TemporalDataFrame({'col1': [i for i in range(1000, 1100)],
                                      'col2': [i for i in range(1100, 1200)]},
                                     name='1_modified',
                                     time_list=['2h' for _ in range(50)] + ['3h' for _ in range(50)])

    # create smaller version
    smaller_TDF = TemporalDataFrame({'col1': [i for i in range(20)],
                                     'col2': [i for i in range(20, 40)]},
                                    name='1_smaller',
                                    time_list=['4h' for _ in range(10)] + ['5h' for _ in range(10)])

    save_path = Path(output_file / 'test_TDF')
    save_path_modified = Path(output_file / 'test_TDF_modified')
    save_path_smaller = Path(output_file / 'test_TDF_smaller')
    save_path_another = Path(output_file / 'test_TDF_another')

    cleanup([save_path, save_path_modified, save_path_smaller, save_path_another])

    # write TDF : not backed
    #   no file path
    with pytest.raises(ValueError) as exc_info:
        TDF.write()

    assert str(exc_info.value) == "A file path must be supplied when write a TemporalDataFrame that is not already " \
                                  "backed on a file."

    #   Path object
    #       Path does not exist
    assert not TDF.is_backed
    TDF.write(save_path)
    check_H5_file(reference_TDF, save_path)
    assert TDF.is_backed

    #       Path exists and data is the same shape
    modified_TDF.write(save_path_modified)
    TDF = get_TDF('1')
    assert not TDF.is_backed
    TDF.write(save_path_modified)
    check_H5_file(reference_TDF, save_path_modified)
    assert TDF.is_backed

    #       Path exists and data is different
    smaller_TDF.write(save_path_smaller)
    TDF = get_TDF('1')
    assert not TDF.is_backed
    TDF.write(save_path_smaller)
    check_H5_file(reference_TDF, save_path_smaller)
    assert TDF.is_backed

    #   H5 file
    TDF = get_TDF('1')
    assert not TDF.is_backed

    with File(save_path_another, 'a') as h5_file:
        TDF.write(h5_file)
        check_H5_file(reference_TDF, save_path_another)
        assert TDF.is_backed

    cleanup([save_path, save_path_modified, save_path_smaller, save_path_another])

    # write TDF : backed
    # TODO

    #   no file path

    #   Path object
    #       Path does not exist

    #       Path exists and data is the same shape

    #       Path exists and data is different

    #   H5 file

    # write TDF : is a view ---------------------------------------------------
    # view of TDF
    TDF = get_TDF('1')

    save_path = Path(output_file / 'test_view_TDF')

    view = TDF[:, range(10, 90), ['col1', 'col4']]

    view.write(save_path)

    check_H5_file(view, save_path)

    cleanup([save_path])

    # view of backed TDF


if __name__ == '__main__':
    test_write()
