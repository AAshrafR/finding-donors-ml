import pandas as pd
from pathlib import Path
from src.utils.config import DATA_PATH

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