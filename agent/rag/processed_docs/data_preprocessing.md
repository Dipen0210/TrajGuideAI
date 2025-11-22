# Data Preprocessing

## CSV Ingestion
- All 9,400 driving CSVs live in `data/raw/`.
- `TrajectoryPreprocessor.load_data()` walks the directory, loads each file with the required columns, and concatenates them into a single DataFrame.
- Missing files or malformed columns raise explicit errors for robustness.

## Feature Selection
Only the core motion + context attributes are retained:
```
[Local_X, Local_Y, v_Vel, v_Acc, Space_Headway, dis_cen,
 i_l, i_r, i_f, dis_l, dis_r, dis_f]
```
These capture vehicle position, velocity, acceleration, lane offsets, indicator flags, and relative distances.

## Cleaning & Normalization
1. Replace infinities with NaN and fill remaining missing values with 0.
2. Fit a `MinMaxScaler` across the entire dataset (stored in `data/processed/scalers.pkl`).
3. Apply scaling to produce a dense `float32` NumPy array ready for sliding-window generation.

## Sliding Windows
- `build_sequences()` iterates through the normalized data and produces overlapping windows of length `window_size` (default 20).
- The prediction target is `(Local_X, Local_Y)` `prediction_horizon` steps ahead (default 1).
- Output shapes:
  - Features: `(num_samples, window_size, num_features)`
  - Targets: `(num_samples, 2)`

## Storage
- Tensors are saved to `data/processed/dataset.npz`.
- The fitted scaler is serialized to `data/processed/scalers.pkl`.
- This ensures both training and inference operate on identical normalization parameters.
