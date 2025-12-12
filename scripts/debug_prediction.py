"""
Debug script to trace the prediction pipeline step by step.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path
from model.inference import TrajectoryInference, _sequence_to_dataframe
from model.dataset import DEFAULT_COLUMNS

# Create a mini test sequence of 20 points going UP in Y
test_sequence = []
for i in range(20):
    test_sequence.append({
        "Local_X": 15.0 + i * 0.1,
        "Local_Y": 1580.0 + i * 5,  # Going UP: 1580 -> 1675
        "v_Vel": 25.0,
        "v_Acc": 0.0,
        "Space_Headway": 50.0,
        "dis_cen": 0.0,
        "i_l": 0,
        "i_r": 0,
        "i_f": 0,
        "dis_l": 100.0,
        "dis_r": 100.0,
        "dis_f": 100.0,
    })

print("=== DEBUG: Prediction Pipeline ===\n")

# Load inference engine
inf = TrajectoryInference()

print(f"1. Scaler Info:")
print(f"   data_min_ (X, Y) = {inf.scaler.data_min_[:2]}")
print(f"   data_max_ (X, Y) = {inf.scaler.data_max_[:2]}")
print(f"   sklearn version in scaler: {getattr(inf.scaler, '__sklearn_version__', 'unknown')}")

print(f"\n2. Test Sequence (last 3 points):")
for pt in test_sequence[-3:]:
    print(f"   X={pt['Local_X']:.2f}, Y={pt['Local_Y']:.2f}")

# Preprocess
df = _sequence_to_dataframe(test_sequence, list(DEFAULT_COLUMNS))
window = df.tail(20).values.astype(np.float32)
print(f"\n3. Window Shape: {window.shape}")
print(f"   Last row (unscaled): X={window[-1, 0]:.2f}, Y={window[-1, 1]:.2f}")

# Scale
scaled_window = inf.scaler.transform(window)
print(f"\n4. After Scaling:")
print(f"   Last row (scaled): X={scaled_window[-1, 0]:.4f}, Y={scaled_window[-1, 1]:.4f}")

# Predict
import torch
tensor_input = torch.from_numpy(scaled_window).unsqueeze(0).float()
with torch.no_grad():
    model_output = inf.model(tensor_input)
    
raw_output = model_output.squeeze(0).numpy()
print(f"\n5. Model Raw Output (normalized):")
print(f"   X_norm={raw_output[0]:.4f}, Y_norm={raw_output[1]:.4f}")

# Inverse transform
real_output = inf._inverse_transform(raw_output)
print(f"\n6. After Inverse Transform (real scale):")
print(f"   Predicted X={real_output[0]:.2f}, Y={real_output[1]:.2f}")

print(f"\n7. *** VERDICT ***")
expected_y = test_sequence[-1]["Local_Y"] + 5  # Approximately (since we were going up)
actual_y = real_output[1]
delta = actual_y - test_sequence[-1]["Local_Y"]
print(f"   Last history Y: {test_sequence[-1]['Local_Y']:.2f}")
print(f"   Predicted Y:    {actual_y:.2f}")
print(f"   Delta:          {delta:+.2f} (expected ~+5)")

if abs(delta) > 50:
    print("\n   ❌ HUGE ERROR! The model or scaler is broken.")
else:
    print("\n   ✅ Prediction is reasonable.")
