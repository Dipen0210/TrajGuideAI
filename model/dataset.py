"""
Data loading and preprocessing utilities for the Agentic Vehicle Trajectory Prediction System.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


DEFAULT_COLUMNS: Sequence[str] = (
    "Local_X",
    "Local_Y",
    "v_Vel",
    "v_Acc",
    "Space_Headway",
    "dis_cen",
    "i_l",
    "i_r",
    "i_f",
    "dis_l",
    "dis_r",
    "dis_f",
)
TARGET_COLUMNS: Tuple[str, str] = ("Local_X", "Local_Y")


def load_all_csv_files(raw_data_dir: Path, columns: Sequence[str]) -> pd.DataFrame:
    """
    Load and concatenate every CSV file from ``raw_data_dir`` using the provided columns.
    """
    csv_files = sorted(raw_data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_data_dir}")

    data_frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            frame = pd.read_csv(csv_path, usecols=columns)
            data_frames.append(frame)
        except ValueError:
            # Column mismatch, load everything then subset if possible.
            frame = pd.read_csv(csv_path)
            missing = [col for col in columns if col not in frame.columns]
            if missing:
                raise ValueError(f"{csv_path} missing required columns: {missing}") from None
            data_frames.append(frame[columns])

    return pd.concat(data_frames, ignore_index=True)


def build_sequences(
    data: np.ndarray,
    window_size: int,
    prediction_horizon: int,
    target_indices: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct sliding windows and prediction targets from feature arrays.
    """
    num_samples = data.shape[0]
    limit = num_samples - window_size - prediction_horizon + 1
    if limit <= 0:
        raise ValueError(
            "Insufficient data for the specified window_size and prediction_horizon."
        )

    features = np.zeros((limit, window_size, data.shape[1]), dtype=np.float32)
    targets = np.zeros((limit, 2), dtype=np.float32)

    for start_idx in range(limit):
        window_end = start_idx + window_size
        target_idx = window_end + prediction_horizon - 1
        features[start_idx] = data[start_idx:window_end]
        targets[start_idx, 0] = data[target_idx, target_indices[0]]
        targets[start_idx, 1] = data[target_idx, target_indices[1]]

    return features, targets


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_raw_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    car_data_dir = repo_root / "car_data"
    if car_data_dir.exists():
        return car_data_dir
    return _project_root() / "data" / "raw"


def _default_processed_path() -> Path:
    return _project_root() / "data" / "processed" / "dataset.npz"


def _default_scaler_path() -> Path:
    return _project_root() / "data" / "processed" / "scalers.pkl"


@dataclass
class TrajectoryPreprocessor:
    raw_data_dir: Path = field(default_factory=_default_raw_data_dir)
    processed_path: Path = field(default_factory=_default_processed_path)
    scaler_path: Path = field(default_factory=_default_scaler_path)
    feature_columns: Sequence[str] = DEFAULT_COLUMNS

    def load_data(self) -> pd.DataFrame:
        return load_all_csv_files(self.raw_data_dir, self.feature_columns)

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return cleaned.reset_index(drop=True)

    def normalize(self, df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values.astype(np.float32))
        return scaled, scaler

    def process(
        self,
        window_size: int = 20,
        prediction_horizon: int = 1,
        save: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = self.clean_data(self.load_data())
        data, scaler = self.normalize(df)

        target_indices = (
            self.feature_columns.index(TARGET_COLUMNS[0]),
            self.feature_columns.index(TARGET_COLUMNS[1]),
        )
        features, targets = build_sequences(data, window_size, prediction_horizon, target_indices)

        if save:
            self.save(features, targets)
            self.save_scaler(scaler)

        return features, targets

    def save(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_path, features=features, targets=targets)

    def save_scaler(self, scaler: MinMaxScaler) -> None:
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, self.scaler_path)


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset backed by preprocessed trajectory sequences.
    """

    def __init__(
        self,
        npz_path: Optional[Path] = Path("vehicle-trajectory-agent/data/processed/dataset.npz"),
        features: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ) -> None:
        if features is None or targets is None:
            if npz_path is None or not Path(npz_path).exists():
                raise FileNotFoundError(
                    "Processed dataset not found. Provide features/targets or a valid npz_path."
                )
            with np.load(npz_path) as data:
                features = data["features"]
                targets = data["targets"]

        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must contain the same number of samples.")

        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def load_processed_dataset(
    npz_path: Path = Path("vehicle-trajectory-agent/data/processed/dataset.npz"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience loader that returns features and targets arrays from the processed dataset file.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {npz_path}.")
    with np.load(npz_path) as data:
        features = data["features"]
        targets = data["targets"]
    return features, targets
