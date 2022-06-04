from typing import Optional, List

import pandas as pd


def get_date_index(
    stock_price_dataframe: pd.DataFrame,
    date_column_name: str = "Date",
) -> pd.DatetimeIndex:
    """Retrieves datetime index given the full stock price dataframe.

    Args:
        stock_price_dataframe: Stock dataframe of all securities.
        date_column_name: Name of the Date column.

    Returns:
        Sorted datetime index with all trading days.

    """
    return pd.to_datetime(
        stock_price_dataframe[date_column_name].unique()
    ).sort_values()


def get_adjustment_series(
    stock_data: pd.DataFrame,
    adjustment_factor_column: str = "AdjustmentFactor",
    adjustment_type: str = "price",
) -> pd.Series:
    """Generates sorted series that enables price or volume adjustment.

    Args:
        stock_data: Stock dataframe of a single security indexed by date.
        adjustment_factor_column: Name of the column that stores adjustment
            factors.
        adjustment_type: Takes 'price' or 'volume' as input, and alters
            adjustment factor series likewise.
    Returns:
        Adjustment factor time series that can be employed to adjust either
        price or volume time series through elementwise multiplication.
    Raises:
        ValueError: if adjustment_type value is neither 'price' or 'volume'.

    """
    if adjustment_type not in ["price", "volume"]:
        raise ValueError(
            "adjustment_type value is neither 'price' nor 'volume'"
        )

    if adjustment_type == "volume":
        return (
            1
            / stock_data.sort_index(ascending=False)[
                f"{adjustment_factor_column}"
            ]
            .cumprod()
            .sort_index()
        )

    return (
        stock_data.sort_index(ascending=False)[f"{adjustment_factor_column}"]
        .cumprod()
        .sort_index()
    )


def get_stock_data(
    security_code: int,
    stock_price_dataframe: pd.DataFrame,
    code_column_name: str = "SecuritiesCode",
    date_column_name: str = "Date",
) -> pd.DataFrame:

    date_index = get_date_index(
        stock_price_dataframe=stock_price_dataframe,
        date_column_name=date_column_name,
    )

    return (
        stock_price_dataframe.query(f"{code_column_name}=={security_code}")
        .sort_values(date_column_name)
        .set_index(date_index)
        .drop([code_column_name, date_column_name], axis=1)
        .copy()
    )


def get_adjusted_stock_data(
    security_code: int,
    stock_price_dataframe: pd.DataFrame,
    code_column_name: str = "SecuritiesCode",
    date_column_name: str = "Date",
    volume_column_name: str = "Volume",
    adjustment_factor_column: str = "AdjustmentFactor",
    price_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generates adjusted prices and volumes for a single security.

    Args:
        security_code: code of the security.
        stock_price_dataframe: Stock dataframe of all securities.
        code_column_name: String with the security code column.
        date_column_name: String with the Date column.
        volume_column_name: String with the Volume of transactions column.
        adjustment_factor_column: String with the column that stores adjustment
            factors.
        price_columns: List with the column names that store price related info,
            that is: Close, Opem, High, Low.

    Returns:
        Adjusted stock price dataframe.
    """

    if not price_columns:
        price_columns = ["High", "Low", "Open", "Close"]

    stock_data = get_stock_data(
        security_code=security_code,
        stock_price_dataframe=stock_price_dataframe,
        code_column_name=code_column_name,
        date_column_name=date_column_name,
    )

    for price_column in price_columns:
        stock_data[price_column] = (
            get_adjustment_series(
                stock_data=stock_data,
                adjustment_factor_column=adjustment_factor_column,
                adjustment_type="price",
            )
            * stock_data[price_column]
        )

    stock_data[volume_column_name] = (
        get_adjustment_series(
            stock_data=stock_data,
            adjustment_factor_column=adjustment_factor_column,
            adjustment_type="volume",
        )
        * stock_data[volume_column_name]
    )

    return stock_data
