# coding: utf-8
# Created on 04/04/2022 17:24
# Author : matteo

# ====================================================
# imports
import numpy as np
from pathlib import Path

from .utils import get_TDF, get_backed_TDF


# ====================================================
# code
def cleanup(path: Path) -> None:
    if path.exists():
        path.unlink(missing_ok=True)


def test_convert():
    # TDF is not backed
    TDF = get_TDF('1')

    #   no time-points
    df = TDF.to_pandas()

    assert np.all(df.values[:, :2] == np.vstack((
        np.concatenate((np.arange(50, 100), np.arange(0, 50))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(df.values[:, 2:] == np.vstack((
        np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str))),
        np.concatenate((np.arange(350, 400).astype(str), np.arange(300, 350).astype(str)))
    )).T)

    #   with time-points
    df = TDF.to_pandas(with_timepoints='time_points')

    assert np.all(df.time_points == ['0.0h' for _ in range(50)] + ['1.0h' for _ in range(50)])
    assert np.all(df.values[:, 1:3] == np.vstack((
        np.concatenate((np.arange(50, 100), np.arange(0, 50))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(df.values[:, 3:] == np.vstack((
        np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str))),
        np.concatenate((np.arange(350, 400).astype(str), np.arange(300, 350).astype(str)))
    )).T)

    # TDF is backed
    input_file = Path(__file__).parent / 'test_convert_TDF'
    cleanup(input_file)

    TDF = get_backed_TDF(input_file, '2')

    #   no time-points
    df = TDF.to_pandas()

    assert np.all(df.values[:, :2] == np.array(range(100)).reshape((50, 2)))
    assert np.all(df.values[:, 2:] == np.array(list(map(str, range(100, 150))), dtype=np.dtype('O')).reshape((50, 1)))

    #   with time-points
    df = TDF.to_pandas(with_timepoints='time_points')

    assert np.all(df.time_points == ['0.0h' for _ in range(25)] + ['1.0h' for _ in range(25)])
    assert np.all(df.values[:, 1:3] == np.array(range(100)).reshape((50, 2)))
    assert np.all(df.values[:, 3:] == np.array(list(map(str, range(100, 150))), dtype=np.dtype('O')).reshape((50, 1)))

    cleanup(input_file)


if __name__ == '__main__':
    test_convert()
