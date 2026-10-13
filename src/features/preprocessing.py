
"""Reusable functions for loading, cleaning, and inspecting the Census dataset."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    PROCESSED_DATA_PATH
)


def load_data(file_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load a CSV dataset into a Pandas DataFrame.

    Args:
        file_path: Path to the CSV file. Defaults to DATA_PATH.

    Returns:
        A DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at: {path}\n"
            "Place census.csv beside these Python files or pass its full path."
        )

    return pd.read_csv(path)


def _check_columns(
    data: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    """Validate that the requested columns exist in the dataset.

    Args:
        data: DataFrame to validate.
        columns: Column names that must be present.

    Returns:
        A list containing the validated column names.

    Raises:
        ValueError: If one or more requested columns are missing.
    """
    columns = list(columns)

    missing = [
        column for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"These columns are missing from the dataset: {missing}"
        )

    return columns


def clean_text(
    data: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Remove leading and trailing whitespace from text columns.

    Args:
        data: DataFrame containing the text columns to clean.
        columns: Columns to clean. If None, all object and string
            columns are selected automatically.

    Returns:
        A cleaned copy of the DataFrame.

    Raises:
        ValueError: If explicitly requested columns do not exist.

    Note:
        The original DataFrame is not modified.
    """
    cleaned_data = data.copy()

    if columns is None:
        columns = cleaned_data.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
    else:
        columns = _check_columns(cleaned_data, columns)

    for column in columns:
        if (
            pd.api.types.is_object_dtype(cleaned_data[column])
            or pd.api.types.is_string_dtype(cleaned_data[column])
        ):
            cleaned_data[column] = cleaned_data[column].str.strip()

    return cleaned_data


def convert_categorical_columns(
    data: pd.DataFrame,
    columns: Iterable[str] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """Convert selected columns to the Pandas category data type.

    Args:
        data: DataFrame containing the columns to convert.
        columns: Column names to convert. Defaults to
            CATEGORICAL_COLUMNS from the configuration file.

    Returns:
        A copy of the DataFrame with the selected columns converted
        to the category data type.

    Raises:
        ValueError: If one or more requested columns do not exist.

    Note:
        This function does not encode categories into numerical values.
    """
    converted_data = data.copy()
    columns = _check_columns(converted_data, columns)

    for column in columns:
        converted_data[column] = converted_data[column].astype("category")

    return converted_data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply basic text cleaning and categorical type conversion.

    Args:
        data: Raw or partially processed DataFrame.

    Returns:
        A cleaned copy of the DataFrame with text whitespace removed
        and configured categorical columns converted to category dtype.

    Raises:
        ValueError: If a configured categorical column is missing.

    Note:
        This function does not remove duplicates or outliers, fill
        missing values, encode categories, or scale numerical features.
    """
    cleaned_data = clean_text(data)
    cleaned_data = convert_categorical_columns(cleaned_data)

    return cleaned_data


def missing_values_summary(data: pd.DataFrame) -> pd.Series:
    """Count missing values in each column.

    Args:
        data: DataFrame to inspect.

    Returns:
        A Series containing the missing-value count for each column,
        sorted from highest to lowest.
    """
    return data.isna().sum().sort_values(ascending=False)


def duplicate_count(data: pd.DataFrame) -> int:
    """Count duplicated rows in the dataset.

    Args:
        data: DataFrame to inspect.

    Returns:
        The number of duplicated rows, excluding the first occurrence
        of each repeated row.
    """
    return int(data.duplicated().sum())


def remove_duplicates(
    data: pd.DataFrame,
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicated rows and reset the DataFrame index.

    Args:
        data: DataFrame from which to remove duplicates.
        keep: Which duplicate occurrence to retain.
            Use 'first' to keep the first occurrence,
            'last' to keep the last occurrence, or
            False to remove all duplicated occurrences.

    Returns:
        A copy of the DataFrame without the selected duplicates
        and with a reset index.

    Raises:
        ValueError: If keep is not 'first', 'last', or False.

    Note:
        This function does not modify the original DataFrame.
        Only use it when removing duplicates is appropriate.
    """
    if keep not in {"first", "last", False}:
        raise ValueError("keep must be 'first', 'last', or False")

    return data.drop_duplicates(keep=keep).reset_index(drop=True).copy()


def numerical_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for numerical features.

    Args:
        data: DataFrame containing the numerical features.

    Returns:
        A DataFrame containing count, mean, standard deviation,
        minimum, quartiles, and maximum for the configured numerical
        columns that exist in the dataset.
    """
    columns = [
        column for column in NUMERICAL_COLUMNS
        if column in data.columns
    ]

    return data[columns].describe().T


def categorical_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for categorical features.

    Args:
        data: DataFrame containing the categorical features.

    Returns:
        A DataFrame containing the count, number of unique categories,
        most frequent value, and its frequency for each configured
        categorical column that exists in the dataset.
    """
    columns = [
        column for column in CATEGORICAL_COLUMNS
        if column in data.columns
    ]

    return data[columns].describe().T


def unique_counts(data: pd.DataFrame) -> pd.Series:
    """Count distinct values in each configured categorical column.

    Args:
        data: DataFrame containing the categorical features.

    Returns:
        A Series containing the number of distinct values per column,
        sorted from highest to lowest.
        Missing values are excluded from the counts.
    """
    columns = [
        column for column in CATEGORICAL_COLUMNS
        if column in data.columns
    ]

    return data[columns].nunique().sort_values(ascending=False)


def split_features_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the input features from the target variable.

    Args:
        data: DataFrame containing both features and target.
        target_column: Name of the column to predict.
            Defaults to TARGET_COLUMN from the configuration file.

    Returns:
        A tuple containing:
            - X: DataFrame with all columns except the target.
            - y: Series containing the target values.

    Raises:
        ValueError: If the target column does not exist.

    Note:
        This function does not perform encoding, scaling, or
        train-test splitting.
    """
    _check_columns(data, [target_column])

    features = data.drop(columns=[target_column]).copy()
    target = data[target_column].copy()

    return features, target

def save_data(data: pd.DataFrame, file_path: str | Path = PROCESSED_DATA_PATH) -> Path:
    """Save a DataFrame to disk and return the path it was written to.

    The output format is chosen from the file extension. The parent folder is
    created automatically if it does not exist.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to save, usually the output of :func:`clean_data`.
    file_path : str or pathlib.Path, default ``PROCESSED_DATA_PATH``
        Destination path ending in ``.csv`` or ``.parquet``.

    Returns
    -------
    pathlib.Path
        The path of the saved file.

    Raises
    ------
    ValueError
        If the file extension is not ``.csv`` or ``.parquet``.

    Notes
    -----
    CSV does not preserve the ``category`` dtype, so columns come back as plain
    text when the file is read again. Parquet keeps the dtypes (requires the
    ``pyarrow`` package).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        data.to_csv(path, index=False)  # index=False avoids an extra index column
    elif path.suffix == ".parquet":
        data.to_parquet(path, index=False)
    else:
        raise ValueError("file_path must end with '.csv' or '.parquet'")

    return path