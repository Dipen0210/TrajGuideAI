# Model Training

## Architecture
- **Model**: `TrajectoryLSTM` with configurable `hidden_size` (default 64), `num_layers` (default 2), optional dropout, and a linear head projecting the final hidden state to `(Local_X, Local_Y)`.
- **Input**: `(batch_size, window_size, num_features)` sequences produced by the preprocessing pipeline.
- **Output**: `(batch_size, 2)` representing the next-step coordinates.

## Hyperparameters
- `window_size`: default 20, aligning with preprocessing window length.
- `prediction_horizon`: fixed at 1 step ahead for maximal responsiveness.
- `batch_size`: default 64 with DataLoader shuffling for the training split.
- `epochs`: default 20, exposed via CLI.
- `learning_rate`: default 1e-3 using Adam optimizer.
- `dropout`: optional for multi-layer regularization.

## Training Loop
1. Load `dataset.npz` via `load_processed_dataset()`.
2. Randomly permute samples and split into 80% train / 20% validation.
3. For each epoch:
   - Call `train_one_epoch()` → forward pass, `MSELoss`, backpropagation, optimizer step.
   - Evaluate with `evaluate()` for validation MSE.
   - Track and persist the best checkpoint when validation loss improves.

## Validation & Checkpointing
- Validation uses the held-out split with the same DataLoader interface (no shuffle).
- Checkpoint path: `model/checkpoints/trajectory_lstm_best.pth`.
- The best validation model is automatically saved, allowing inference scripts to load the latest stable weights.
