"""
Data loading and preprocessing for UCI HAR dataset.

This module handles downloading, loading, preprocessing, and splitting
the UCI Human Activity Recognition Using Smartphones dataset.

Protocol:
    - All preprocessing (scaling, encoding) is fit on training data only.
    - Subject-aware splitting: no person leaks across train/test.
    - Test set is sealed for final comparison only.
    - Random seeds are fixed for reproducibility.
"""

import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Constants
# ============================================================================

RANDOM_SEEDS = [42, 123, 456]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    # "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip" # this url is for a mirror version coursera published 
)

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

NUM_CLASSES = len(ACTIVITY_LABELS)


# ============================================================================
# Download
# ============================================================================


def download_dataset(force: bool = False) -> Path:
    """Download the UCI HAR dataset if not already present.

    Parameters
    ----------
    force : bool
        If True, re-download even if files exist.

    Returns
    -------
    Path
        Path to the extracted dataset directory.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "UCI_HAR_Dataset.zip"
    extract_dir = RAW_DIR / "UCI HAR Dataset"

    if extract_dir.exists() and not force:
        print(f"Dataset already exists at {extract_dir}")
        return extract_dir

    print(f"Downloading UCI HAR dataset from {UCI_HAR_URL}...")
    urllib.request.urlretrieve(UCI_HAR_URL, zip_path)

    print(f"Extracting to {RAW_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    zip_path.unlink()  # Clean up zip
    print("Download complete.")
    return extract_dir


# ============================================================================
# Loading
# ============================================================================


def _load_file(filepath: Path) -> np.ndarray:
    """Load a single whitespace-delimited file into a NumPy array."""
    return np.loadtxt(filepath)


def load_raw_data(dataset_dir: Path | None = None) -> dict:
    """Load the raw UCI HAR train/test data.

    Returns
    -------
    dict with keys:
        X_train, y_train, subject_train,
        X_test, y_test, subject_test
    """
    if dataset_dir is None:
        dataset_dir = RAW_DIR / "UCI HAR Dataset"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. "
            "Run `uv run python src/data.py --download` first."
        )

    data = {}
    for split in ["train", "test"]:
        split_dir = dataset_dir / split
        data[f"X_{split}"] = _load_file(split_dir / f"X_{split}.txt")
        data[f"y_{split}"] = _load_file(split_dir / f"y_{split}.txt").astype(int)
        data[f"subject_{split}"] = _load_file(
            split_dir / f"subject_{split}.txt"
        ).astype(int)

    return data


# ============================================================================
# Preprocessing
# ============================================================================


def preprocess(
    X_train: np.ndarray,
    X_test: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Preprocess features - fit on train, transform both.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test : np.ndarray, shape (n_test, n_features)
    normalize : bool
        Whether to apply z-score normalization.

    Returns
    -------
    X_train_proc, X_test_proc, preprocessing_info
    """
    info = {"normalize": normalize}

    if normalize:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # Prevent division by zero

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        info["mean"] = mean
        info["std"] = std

    return X_train, X_test, info


# ============================================================================
# Convenience
# ============================================================================


def get_data(normalize: bool = True) -> dict:
    """Load, preprocess, and return the dataset ready for experiments.

    Returns
    -------
    dict with keys:
        X_train, y_train, subject_train,
        X_test, y_test, subject_test,
        preprocessing_info
    """
    raw = load_raw_data()
    X_train, X_test, info = preprocess(
        raw["X_train"], raw["X_test"], normalize=normalize
    )

    return {
        "X_train": X_train,
        "y_train": raw["y_train"].ravel(),
        "subject_train": raw["subject_train"].ravel(),
        "X_test": X_test,
        "y_test": raw["y_test"].ravel(),
        "subject_test": raw["subject_test"].ravel(),
        "preprocessing_info": info,
        "activity_labels": ACTIVITY_LABELS,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UCI HAR Dataset Manager")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--info", action="store_true", help="Print dataset info")
    args = parser.parse_args()

    if args.download:
        download_dataset()

    if args.info or not args.download:
        try:
            data = get_data(normalize=False)
            print(f"X_train shape: {data['X_train'].shape}")
            print(f"y_train shape: {data['y_train'].shape}")
            print(f"X_test shape:  {data['X_test'].shape}")
            print(f"y_test shape:  {data['y_test'].shape}")
            print(f"Train subjects: {np.unique(data['subject_train'])}")
            print(f"Test subjects:  {np.unique(data['subject_test'])}")
            print(f"Activity labels: {ACTIVITY_LABELS}")
            print(f"Class distribution (train): {np.bincount(data['y_train'])[1:]}")
            print(f"Class distribution (test):  {np.bincount(data['y_test'])[1:]}")
        except FileNotFoundError as e:
            print(e)
