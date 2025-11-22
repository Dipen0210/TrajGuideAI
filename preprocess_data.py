from model.dataset import TrajectoryPreprocessor

def main():
    print("Starting data preprocessing...")
    preprocessor = TrajectoryPreprocessor()
    # Using default parameters as defined in the class: window_size=20, prediction_horizon=1
    features, targets = preprocessor.process()
    print(f"Preprocessing complete. Features shape: {features.shape}, Targets shape: {targets.shape}")
    print(f"Data saved to {preprocessor.processed_path}")
    print(f"Scaler saved to {preprocessor.scaler_path}")

if __name__ == "__main__":
    main()
