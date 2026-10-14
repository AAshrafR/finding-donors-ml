"""Project configuration: paths, column names, and feature groups."""

from pathlib import Path

# Folder that contains this file: <project>/src/features
BASE_DIR = Path(__file__).resolve().parent

# Project root = two levels up from src/features
PROJECT_ROOT = BASE_DIR.parents[2]

# Paths are built from the project root, so they work on any machine
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "census.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "census_cleaned.csv"
# Target column we want to predict.
TARGET_COLUMN = "income"


# Categorical columns (including the target label).
CATEGORICAL_COLUMNS = [
    "workclass",
    "education_level",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]

# Numerical columns.
NUMERICAL_COLUMNS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
