"""
Visualization helpers for the Streamlit app.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(sequence: List[Dict[str, float]], prediction: Optional[Dict[str, float]] = None):
    """
    Plot Local_X vs Local_Y for the provided sequence and mark the predicted point if supplied.
    """
    if not sequence:
        raise ValueError("Empty sequence provided for plotting.")

    xs = [item["Local_X"] for item in sequence]
    ys = [item["Local_Y"] for item in sequence]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", linestyle="-", color="#1f77b4", label="Trajectory")
    ax.set_xlabel("Local_X")
    ax.set_ylabel("Local_Y")
    ax.set_title("Vehicle Trajectory")
    ax.grid(True, linestyle="--", alpha=0.4)

    if prediction:
        ax.scatter(
            prediction.get("predicted_local_x"),
            prediction.get("predicted_local_y"),
            color="red",
            label="Predicted Point",
            zorder=5,
        )
        ax.legend()

    fig.tight_layout()
    return fig


def plot_risk_meter(risk_score: float):
    """
    Display a gauge-like representation for risk score between 0 and 1.
    """
    risk_score = max(0.0, min(1.0, risk_score))

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")

    theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=2)
    ax.fill_between(np.cos(theta), 0, np.sin(theta), color="#f0f0f0", alpha=0.5)

    needle_angle = (-np.pi / 2) + risk_score * np.pi
    ax.plot(
        [0, np.cos(needle_angle)],
        [0, np.sin(needle_angle)],
        color="red",
        linewidth=3,
    )
    ax.text(0, -0.2, f"Risk: {risk_score:.2f}", ha="center", fontsize=12, weight="bold")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.2, 1.1)

    fig.tight_layout()
    return fig
