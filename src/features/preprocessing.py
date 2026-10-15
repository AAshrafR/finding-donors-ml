
"""Reusable functions for loading, cleaning, and inspecting the Census dataset."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    PROCESSED_DATA_PATH
)




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
# =============================================================================
# Preprocessing Functions
# =============================================================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the data.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.
    """
    data = pd.read_csv(data_path)
    return data


def split_target_and_features(
    data: pd.DataFrame,
    target_column: str = 'income'
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Split the target column from the feature columns.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataframe.
    target_column : str, optional
        Name of the target column, default is 'income'.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        - income_raw: Series containing the target values.
        - features_raw: DataFrame containing only the features.
    """
    income_raw = data[target_column]
    features_raw = data.drop(target_column, axis=1)
    return income_raw, features_raw


def handle_skewed_features(
    features_raw: pd.DataFrame,
    skewed: List[str] = None
) -> pd.DataFrame:
    """
    Handle skewed features using logarithmic transformation.

    Parameters
    ----------
    features_raw : pd.DataFrame
        DataFrame containing the raw features.
    skewed : List[str], optional
        List of skewed column names.
        Default is ['capital-gain', 'capital-loss'].

    Returns
    -------
    pd.DataFrame
        New dataframe after applying log transformation on the specified columns.
    """
    if skewed is None:
        skewed = ['capital-gain', 'capital-loss']

    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    return features_log_transformed


def normalize_features(
    features_log_transformed: pd.DataFrame,
    numerical: List[str] = None
) -> pd.DataFrame:
    """
    Normalize numerical features using MinMaxScaler to the range (0, 1).

    Parameters
    ----------
    features_log_transformed : pd.DataFrame
        DataFrame after log transformation.
    numerical : List[str], optional
        List of numerical column names to be normalized.
        Default is ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'].

    Returns
    -------
    pd.DataFrame
        New dataframe after normalizing the specified numerical columns.
    """
    if numerical is None:
        numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    scaler = MinMaxScaler()  # default=(0, 1)
    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(
        features_log_transformed[numerical]
    )
    return features_log_minmax_transform


def encode_features_and_target(
    features_log_minmax_transform: pd.DataFrame,
    income_raw: pd.Series,
    positive_label: str = '>50K'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply one-hot encoding to features and encode the target column into numerical values.

    Parameters
    ----------
    features_log_minmax_transform : pd.DataFrame
        DataFrame after normalization.
    income_raw : pd.Series
        Original series containing the textual target values.
    positive_label : str, optional
        The textual value representing the positive class, default is '>50K'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        - features_final: DataFrame after applying one-hot encoding.
        - income: Series containing the numerically encoded target values (1 or 0).
    """
    features_final = pd.get_dummies(features_log_minmax_transform)
    income = income_raw.apply(lambda x: 1 if x == positive_label else 0)

    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))

    return features_final, income


def split_train_test(
    features_final: pd.DataFrame,
    income: pd.Series,
    test_size: float = 0.2,
    random_state: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    features_final : pd.DataFrame
        Final DataFrame containing the features.
    income : pd.Series
        Series containing the encoded target values.
    test_size : float, optional
        Proportion of the testing data, default is 0.2.
    random_state : int, optional
        Random seed for reproducibility, default is 0.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        - X_train: Training features.
        - X_test: Testing features.
        - y_train: Training targets.
        - y_test: Testing targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features_final,
        income,
        test_size=test_size,
        random_state=random_state
    )

    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    return X_train, X_test, y_train, y_test


def preprocessing_data(
    data_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Execute the full preprocessing pipeline on the data.

    Steps:
        1. Load the data.
        2. Split target from features.
        3. Handle skewed features.
        4. Normalize numerical features.
        5. Encode features and target.
        6. Split data into training and testing sets.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        - X_train: Training features.
        - X_test: Testing features.
        - y_train: Training targets.
        - y_test: Testing targets.
    """
    data = load_data(data_path)
    income_raw, features_raw = split_target_and_features(data)
    features_log_transformed = handle_skewed_features(features_raw)
    features_log_minmax_transform = normalize_features(features_log_transformed)
    features_final, income = encode_features_and_target(
        features_log_minmax_transform, income_raw
    )
    X_train, X_test, y_train, y_test = split_train_test(features_final, income)

    return X_train, X_test, y_train, y_test
