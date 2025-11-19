"""
LangChain tool for running trajectory predictions via the trained LSTM model.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from langchain.tools import tool

from model.inference import TrajectoryInference


_INFERENCE = TrajectoryInference()


def _sequence_to_dataframe(sequence: List) -> pd.DataFrame:
    """
    Convert supported sequence representations into a DataFrame that matches the model schema.
    """
    if not sequence:
        raise ValueError("Input sequence is empty.")

    first = sequence[0]
    columns = _INFERENCE.feature_columns

    if isinstance(first, dict):
        df = pd.DataFrame(sequence)
    else:
        if len(first) != len(columns):
            raise ValueError(
                f"Sequence elements must contain {len(columns)} values matching the feature columns."
            )
        df = pd.DataFrame(sequence, columns=columns)

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Sequence missing required columns: {missing}")

    return df[columns]


@tool("predict_trajectory")
def predict_trajectory(input_sequence: List) -> dict:
    """
    Receives a raw time-series sequence of dictionaries or lists.
    Preprocess using TrajectoryInference and return the next-step prediction.
    """

    df = _sequence_to_dataframe(input_sequence)
    return _INFERENCE.predict(df)
