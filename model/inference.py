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
    hidden_size: Optional[int],
    output_size: Optional[int],
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> TrajectoryLSTM:
    """
    Instantiate the model and load weights from disk.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    inferred_hidden_size = (
        state_dict["lstm.weight_ih_l0"].shape[0] // 4 if hidden_size is None else hidden_size
    )
    inferred_output_size = state_dict["head.weight"].shape[0] if output_size is None else output_size

    # Guardrail: make sure user-specified dims match the checkpoint to avoid silent shape errors.
    if hidden_size is not None and hidden_size != inferred_hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} does not match checkpoint hidden_size={inferred_hidden_size}"
        )
    if output_size is not None and output_size != inferred_output_size:
        raise ValueError(
            f"output_size={output_size} does not match checkpoint output_size={inferred_output_size}"
        )

    model = TrajectoryLSTM(
        input_size=input_size,
        hidden_size=inferred_hidden_size,
        output_size=inferred_output_size,
        num_layers=num_layers,
        dropout=dropout,
    )
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
        hidden_size: Optional[int] = None,
        output_size: Optional[int] = None,
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
            output_size=output_size,  # Defaults to predicting full state vector
            num_layers=num_layers,
            dropout=dropout,
            device=self.device,
        )

    def preprocess(self, sequence: Iterable) -> np.ndarray:
        """
        Normalize the latest window of the provided raw sequence.
        """
        return preprocess_sequence(sequence, self.feature_columns, self.scaler, self.window_size)

    def _inverse_transform(self, prediction: np.ndarray, target_indices: list = None) -> np.ndarray:
        """
        Map normalized predictions back to the original scale.
        
        If the model outputs fewer features than the scaler expects (legacy 2-output model),
        we only inverse-transform those specific columns.
        """
        num_features = self.scaler.n_features_in_
        
        # If prediction matches scaler size, do full inverse transform
        if prediction.shape[0] == num_features:
            return self.scaler.inverse_transform(prediction.reshape(1, -1)).flatten()
        
        # Legacy model: prediction is only for specific target columns (e.g., Local_X, Local_Y)
        # We need to inverse transform only those columns
        if target_indices is None:
            # Assume first N columns where N = prediction size
            target_indices = list(range(prediction.shape[0]))
        
        # Create a dummy full-size array, fill in the predictions, inverse transform, then extract
        dummy = np.zeros((1, num_features), dtype=np.float32)
        
        # Use the scaler's min/max values to place predictions correctly
        for i, idx in enumerate(target_indices):
            dummy[0, idx] = prediction[i]
        
        # Inverse transform the full array
        full_inverse = self.scaler.inverse_transform(dummy).flatten()
        
        # Return only the target columns
        return np.array([full_inverse[idx] for idx in target_indices])

    def predict(self, sequence: Iterable, steps: int = 1) -> dict:
        """
        Predict the next `steps` trajectory points.
        
        Handles both legacy 2-output models (Local_X, Local_Y) and full 12-feature models.
        """
        # 1. Preprocess the initial window
        df = _sequence_to_dataframe(sequence, self.feature_columns)
        if len(df) < self.window_size:
            raise ValueError(f"Sequence length {len(df)} < window_size={self.window_size}")

        current_window_df = df.tail(self.window_size).copy()
        current_window_vals = current_window_df[self.feature_columns].values.astype(np.float32)
        
        # Determine model output size and target columns
        # Check the model's output layer to determine what it predicts
        output_size = self.model.head.out_features
        
        # For legacy 2-output models, targets are Local_X, Local_Y (indices 0, 1)
        if output_size == 2:
            target_columns = ["Local_X", "Local_Y"]
            target_indices = [0, 1]  # First two columns
        else:
            target_columns = list(self.feature_columns)
            target_indices = list(range(len(self.feature_columns)))
        
        predictions = []

        for _ in range(steps):
            # Scale current window
            scaled_window = self.scaler.transform(current_window_vals)
            
            # Prepare tensor
            tensor_input = torch.from_numpy(scaled_window).unsqueeze(0).to(self.device).float()
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                output = self.model(tensor_input)
            
            # Inverse transform output
            norm_pred = output.squeeze(0).cpu().numpy()
            real_pred = self._inverse_transform(norm_pred, target_indices)
            
            # Package result
            pred_dict = {}
            for col, val in zip(target_columns, real_pred):
                pred_dict[col] = float(val)
            
            # Add convenient access keys for compatibility
            if "Local_X" in pred_dict:
                pred_dict["predicted_local_x"] = pred_dict["Local_X"]
            if "Local_Y" in pred_dict:
                pred_dict["predicted_local_y"] = pred_dict["Local_Y"]
            
            predictions.append(pred_dict)
            
            # Update window for next step:
            # For legacy model, we only update X and Y; keep other features unchanged
            if output_size == 2:
                new_row = current_window_vals[-1].copy()
                new_row[0] = real_pred[0]  # Local_X
                new_row[1] = real_pred[1]  # Local_Y
            else:
                new_row = real_pred
            
            # Slide window: drop first, append new
            current_window_vals = np.vstack([current_window_vals[1:], new_row])
            
        return {"trajectory": predictions}
