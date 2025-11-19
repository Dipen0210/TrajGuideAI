"""
Training script for the TrajectoryLSTM model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model.dataset import load_processed_dataset
from model.lstm_model import TrajectoryLSTM


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data/processed/dataset.npz"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints/trajectory_lstm_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TrajectoryLSTM model.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--train_split", type=float, default=0.8)
    return parser.parse_args()


def split_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    num_samples = features.shape[0]
    permutation = np.random.permutation(num_samples)
    train_size = int(num_samples * train_ratio)
    train_idx = permutation[:train_size]
    val_idx = permutation[train_size:]
    if val_idx.size == 0:
        val_idx = permutation[-1:]
        train_idx = permutation[:-1]

    train_features = features[train_idx]
    train_targets = targets[train_idx]
    val_features = features[val_idx]
    val_targets = targets[val_idx]
    return (train_features, train_targets), (val_features, val_targets)


def build_dataloader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensor_x = torch.from_numpy(features.astype(np.float32))
    tensor_y = torch.from_numpy(targets.astype(np.float32))
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: TrajectoryLSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(
    model: TrajectoryLSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


def save_checkpoint(model: TrajectoryLSTM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    args = parse_args()
    print("Hyperparameters:", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, targets = load_processed_dataset(PROCESSED_PATH)
    if features.shape[1] != args.window_size:
        raise ValueError(
            f"Window size mismatch. Processed data has {features.shape[1]}, "
            f"but --window_size was set to {args.window_size}."
        )

    num_features = features.shape[2]
    (train_features, train_targets), (val_features, val_targets) = split_dataset(
        features, targets, args.train_split
    )

    train_loader = build_dataloader(train_features, train_targets, args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_features, val_targets, args.batch_size, shuffle=False)

    model = TrajectoryLSTM(
        input_size=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, CHECKPOINT_PATH)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
