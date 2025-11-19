"""
Inference utilities for the TrajectoryLSTM model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from model.dataset import DEFAULT_COLUMNS
from model.lstm_model import TrajectoryLSTM


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/trajectory_lstm_best.pth"
DEFAULT_SCALER_PATH = BASE_DIR / "data/processed/scalers.pkl"


def load_model(
    checkpoint_path: Path,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> TrajectoryLSTM:
    """
    Instantiate the model and load weights from disk.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = TrajectoryLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_scalers(scaler_path: Path) -> MinMaxScaler:
    """
    Load the MinMaxScaler used during preprocessing.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler: MinMaxScaler = joblib.load(scaler_path)
    return scaler


def preprocess_sequence(
    sequence: Iterable,
    feature_columns: Sequence[str],
    scaler: MinMaxScaler,
    window_size: int,
) -> np.ndarray:
    """
    Convert a raw sequence to a normalized numpy array with shape (window_size, num_features).
    """
    df = _sequence_to_dataframe(sequence, feature_columns)
    df = df[feature_columns]

    if len(df) < window_size:
        raise ValueError(f"Sequence length {len(df)} is shorter than window_size={window_size}.")

    window = df.tail(window_size)
    normalized = scaler.transform(window.astype(np.float32))
    return normalized.astype(np.float32)


def predict(model: TrajectoryLSTM, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Run a forward pass in evaluation mode and return the model output.
    """
    model.eval()
    with torch.no_grad():
        return model(inputs.to(device))


def _sequence_to_dataframe(sequence: Iterable, feature_columns: Sequence[str]) -> pd.DataFrame:
    """
    Convert supported raw sequence types into a pandas DataFrame.
    """
    if isinstance(sequence, pd.DataFrame):
        return sequence.copy()

    if isinstance(sequence, np.ndarray):
        if sequence.ndim != 2 or sequence.shape[1] != len(feature_columns):
            raise ValueError(
                f"Numpy array must have shape (timesteps, {len(feature_columns)}) but got {sequence.shape}."
            )
        return pd.DataFrame(sequence, columns=feature_columns)

    # Assume iterable of dicts or iterables.
    sequence_list = list(sequence)
    if not sequence_list:
        raise ValueError("Empty sequence provided for inference.")

    first_elem = sequence_list[0]
    if isinstance(first_elem, dict):
        df = pd.DataFrame(sequence_list)
    else:
        df = pd.DataFrame(sequence_list, columns=feature_columns)

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input sequence missing columns: {missing}")

    return df


class TrajectoryInference:
    """
    High-level interface to run LSTM trajectory predictions on new sequences.
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        scaler_path: Path = DEFAULT_SCALER_PATH,
        feature_columns: Optional[Sequence[str]] = None,
        window_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        self.feature_columns = list(feature_columns) if feature_columns else list(DEFAULT_COLUMNS)
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = load_scalers(scaler_path)
        self.model = load_model(
            checkpoint_path=checkpoint_path,
            input_size=len(self.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=self.device,
        )
        self.target_indices = (
            self.feature_columns.index("Local_X"),
            self.feature_columns.index("Local_Y"),
        )

    def preprocess(self, sequence: Iterable) -> np.ndarray:
        """
        Normalize the latest window of the provided raw sequence.
        """
        return preprocess_sequence(sequence, self.feature_columns, self.scaler, self.window_size)

    def _inverse_transform(self, prediction: np.ndarray) -> np.ndarray:
        """
        Map normalized predictions back to the original scale.
        """
        data_min = self.scaler.data_min_[list(self.target_indices)]
        data_max = self.scaler.data_max_[list(self.target_indices)]
        data_range = data_max - data_min
        return prediction * data_range + data_min

    def predict(self, sequence: Iterable) -> dict:
        """
        Predict the next (Local_X, Local_Y) values for the provided sequence.
        """
        normalized_window = self.preprocess(sequence)
        tensor_input = torch.from_numpy(normalized_window).unsqueeze(0).to(self.device)
        outputs = predict(self.model, tensor_input, self.device)
        normalized_prediction = outputs.squeeze(0).cpu().numpy()
        real_values = self._inverse_transform(normalized_prediction)
        return {
            "predicted_local_x": float(real_values[0]),
            "predicted_local_y": float(real_values[1]),
        }
