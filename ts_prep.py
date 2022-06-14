"""
This file contains functions that will be useful in trating and preparing
time series data before feeding it into other algorithms.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


class SeriesToSequences:
    def __init__(self, time_dim: int, sequence_columns: List[str]) -> None:
        pass


def sequence_to_vectors(
    series: Union[np.ndarray, pd.Series],
    time_dim: int,
    output_dim: int = 1,
    prefix: str = "x",
    target_name: Union[List[str], str] = "target",
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Generates feature vectors from sequence data. Generates both input
    time feature vector and target feature vector. In case the output dimension
    is zero, no target will be provided. In case it is greater than one,
    the parameter target_name will be treated as the prefix in case it is
    provided as a string.
    """

    if isinstance(target_name, list):
        if len(target_name) != output_dim:
            raise IndexError(
                "Output dimension and target name list length do not match."
            )

    total_dim = time_dim + output_dim

    target_name = [] if output_dim == 0 else target_name
    if output_dim >= 1:
        if isinstance(target_name, str):
            target_name = [
                f"{target_name}{j}" if output_dim != 1 else target_name
                for j in range(output_dim)
            ]

    col_names = [f"{prefix}{j}" for j in range(time_dim)] + target_name

    data = pd.DataFrame([], columns=col_names)
    for i in range(series.size - total_dim + 1):
        data = data.append(
            pd.Series(series[i : i + total_dim].to_list(), index=col_names),
            ignore_index=True,
        )

    if output_dim:
        X, y = data[col_names[:-output_dim]], data[target_name]
        return X, y

    return data


def hurst_exponent(series: pd.Series) -> float:
    """
    Calculates the Hurst Exponent from a time series.
    """
    delta_b = []
    time_windows = list(range(1, int(np.sqrt(series.shape[0])), 1))
    for time_window in time_windows:
        delta_b.append((series - series.shift(time_window)).abs().mean())
    hurst_exponent_hat = np.polyfit(
        x=np.log(time_windows), y=np.log(delta_b), deg=1
    )[0]
    return hurst_exponent_hat
