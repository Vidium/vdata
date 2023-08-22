from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import pandas as pd

import vdata
from vdata.IO.logger import generalLogger
from vdata.tdf import TemporalDataFrame
from vdata.timepoint import TimePoint
from vdata.utils import spacer


def _get_time_col_name(
    time_list: Sequence[str | TimePoint] | Literal["*"] | None,
    time_col_name: str | None,
    metadata: dict[str, Any] | None,
    *metadata_keys: str,
) -> str | None:
    if time_list is not None or time_col_name is not None:
        return time_col_name

    if metadata is None:
        return None

    for key in metadata_keys:
        metadata = cast(dict[str, Any], metadata[key])

    return str(metadata["timepoints_column_name"])


def read_from_csv(
    path: str | Path,
    time_list: Sequence[str | TimePoint] | Literal["*"] | None = None,
    time_col_name: str | None = None,
    name: str = "",
) -> vdata.VData:
    """
    Function for reading data from csv datasets and building a VData object.

    Args:
        directory: a path to a directory containing csv datasets.
            The directory should have the format, for any combination of the following datasets :
                ⊦ layers
                    ⊦ <...>.csv
                ⊦ obsm
                    ⊦ <...>.csv
                ⊦ obsp
                    ⊦ <...>.csv
                ⊦ varm
                    ⊦ <...>.csv
                ⊦ varp
                    ⊦ <...>.csv
                ⊦ obs.csv
                ⊦ timepoints.csv
                ⊦ var.csv
        time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
        time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
            given. This column will be used as the time data.
        name: an optional name for the loaded VData object.
    """
    parsed_directory = Path(path).expanduser()

    # make sure the path exists
    if not parsed_directory.exists():
        raise ValueError(f"The path {parsed_directory} does not exist.")

    # load metadata if possible
    metadata = None

    if (parsed_directory / ".metadata.json").is_file():
        with open(parsed_directory / ".metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)

    obs: TemporalDataFrame | None = None
    data_df: dict[str, pd.DataFrame] = {}
    data_dicts: dict[str, dict[str, TemporalDataFrame]] = {}
    df_dicts: dict[str, dict[str, pd.DataFrame]] = {}

    # import the data
    for f in sorted(parsed_directory.iterdir()):
        if f.name == ".metadata.json":
            continue

        generalLogger.info(lambda: f"Got key : '{f.name}'.")

        if f.suffix == ".csv":
            if f.name in ("var.csv", "timepoints.csv"):
                generalLogger.info(lambda: f"{spacer(1)}Reading pandas DataFrame '{f.name[:-4]}'.")
                data_df[f.name[:-4]] = pd.read_csv(parsed_directory / f.name, index_col=0)

            elif f.name == "obs.csv":
                generalLogger.info(lambda: f"{spacer(1)}Reading TemporalDataFrame '{f.name[:-4]}'.")

                obs = TemporalDataFrame.read_from_csv(
                    parsed_directory / f.name,
                    timepoints=time_list,
                    time_col_name=_get_time_col_name(time_list, time_col_name, metadata, "obs"),
                )

        else:
            generalLogger.info(lambda: f"{spacer(1)}Reading group '{f.name}'.")

            if f.name in ("layers", "obsm"):
                dataset_dict: dict[str, TemporalDataFrame] = {}

                for dataset in sorted((parsed_directory / f.name).iterdir()):
                    generalLogger.info(lambda: f"{spacer(2)} Reading TemporalDataFrame {dataset.name[:-4]}")

                    dataset_dict[dataset.name[:-4]] = TemporalDataFrame.read_from_csv(
                        parsed_directory / f.name / dataset.name,
                        timepoints=time_list,
                        time_col_name=_get_time_col_name(time_list, time_col_name, metadata, f.name, dataset.name[:-4]),
                    )

                data_dicts[f.name] = dataset_dict

            elif f.name in ("obsp", "varm", "varp"):
                df_dict: dict[str, pd.DataFrame] = {}

                for dataset in sorted((parsed_directory / f.name).iterdir()):
                    generalLogger.info(lambda: f"{spacer(2)} Reading pandas DataFrame {dataset.name}")
                    df_dict[dataset.name[:-4]] = pd.read_csv(parsed_directory / f.name, index_col=0)

                df_dicts[f.name] = df_dict

    return vdata.VData(
        data_dicts.get("layers", None),
        obs,
        data_dicts.get("obsm", None),
        df_dicts.get("obsp", None),
        data_df.get("var", None),
        df_dicts.get("varm", None),
        df_dicts.get("varp", None),
        data_df.get("timepoints", None),
        name=name,
    )
