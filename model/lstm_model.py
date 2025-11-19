"""
PyTorch LSTM model for vehicle trajectory prediction.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TrajectoryLSTM(nn.Module):
    """
    Multi-layer LSTM followed by a linear projection to predict next (Local_X, Local_Y).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch_size, seq_len, num_features)
        Returns:
            Predictions of shape (batch_size, 2)
        """
        outputs, _ = self.lstm(x)
        last_hidden = outputs[:, -1, :]
        return self.head(last_hidden)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
