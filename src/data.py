"""
Data loading and preprocessing utilities.

This module handles loading raw transaction data, cleaning, and saving
processed datasets for use in notebooks and production code.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from .config import (
    DATA_RAW,
    DATA_PROCESSED,
)

# ============================================================================
# Data Processing Constants
# ============================================================================

# Raw data file names
RAW_TRANSACTIONS_FILE_1 = "transactions_1.csv"
RAW_TRANSACTIONS_FILE_2 = "transactions_2.csv"

# Processed data file name
PROCESSED_TRANSACTIONS_FILE = "transactions_cleaned.csv"

# Product names to consolidate into "Other"
ODD_PRODUCTS = ["Not a make", "Undefined", "├ÅTS"]
OTHER_PRODUCT_LABEL = "Other"


def load_raw_transactions(
    data_dir: Optional[Path] = None,
    file1: Optional[str] = None,
    file2: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and combine raw transaction CSV files.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing raw data files. If None, uses DATA_RAW from config.
    file1 : str, optional
        Name of first transaction file. If None, uses RAW_TRANSACTIONS_FILE_1 from config.
    file2 : str, optional
        Name of second transaction file. If None, uses RAW_TRANSACTIONS_FILE_2 from config.
    
    Returns
    -------
    pd.DataFrame
        Combined raw transaction data.
    """
    if data_dir is None:
        data_dir = DATA_RAW
    
    if file1 is None:
        file1 = RAW_TRANSACTIONS_FILE_1
    
    if file2 is None:
        file2 = RAW_TRANSACTIONS_FILE_2
    
    # Load both files, handling the unnamed index column
    df1 = pd.read_csv(data_dir / file1, index_col=0)
    df2 = pd.read_csv(data_dir / file2, index_col=0)
    
    # Combine into single dataframe
    df = pd.concat([df1, df2], ignore_index=True)
    
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data: convert dates, remove duplicates, clean product names.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe with columns: customer_id, product_id, date
    
    Returns
    -------
    pd.DataFrame
        Cleaned transaction dataframe.
    """
    df = df.copy()
    
    # Convert date column to datetime64[ns] tz-naive
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    
    # Remove duplicates (keep first occurrence)
    initial_count = len(df)
    df = df.drop_duplicates(keep='first')
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"Removed {removed} duplicate rows (kept first occurrence)")
    
    # Clean product names: merge rare/non-meaningful labels into "Other"
    replacement_map = {p: OTHER_PRODUCT_LABEL for p in ODD_PRODUCTS}
    df['product_id'] = df['product_id'].replace(replacement_map)
    
    # Sort by date for consistent ordering
    df = df.sort_values('date').reset_index(drop=True)
    
    # Assertions
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].dt.tz is None
    
    return df


def load_and_process_transactions(
    data_raw_dir: Optional[Path] = None,
    data_processed_dir: Optional[Path] = None,
    output_file: Optional[str] = None,
    force_reprocess: bool = False
) -> pd.DataFrame:
    """
    Load raw transactions, clean them, and optionally save processed data.
    
    This is the main entry point for data loading. It will:
    1. Load raw CSV files
    2. Clean the data (dates, duplicates, product names)
    3. Save to processed directory if file doesn't exist or force_reprocess=True
    4. Return the cleaned dataframe
    
    Parameters
    ----------
    data_raw_dir : Path, optional
        Directory containing raw data files. If None, uses DATA_RAW from config.
    data_processed_dir : Path, optional
        Directory to save processed data. If None, uses DATA_PROCESSED from config.
    output_file : str, optional
        Name of output file (CSV format). If None, uses PROCESSED_TRANSACTIONS_FILE from config.
    force_reprocess : bool
        If True, reprocess even if processed file exists.
    
    Returns
    -------
    pd.DataFrame
        Cleaned transaction dataframe.
    """
    # Set up paths using config defaults
    if data_raw_dir is None:
        data_raw_dir = DATA_RAW
    
    if data_processed_dir is None:
        data_processed_dir = DATA_PROCESSED
    
    if output_file is None:
        output_file = PROCESSED_TRANSACTIONS_FILE
    
    # Create processed directory if it doesn't exist
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_processed_dir / output_file
    
    # Check if processed file exists and we don't want to force reprocess
    if output_path.exists() and not force_reprocess:
        print(f"Loading processed data from {output_path}")
        df = pd.read_csv(output_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
        # Assertions
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].dt.tz is None
        return df
    
    # Load and process raw data
    print("Loading raw transaction data...")
    df = load_raw_transactions(data_raw_dir)
    print(f"Loaded {len(df):,} raw transactions")
    
    print("Cleaning transaction data...")
    df = clean_transactions(df)
    print(f"Cleaned data: {len(df):,} transactions")
    
    # Save processed data
    print(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    
    return df
