"""Normalize pixels and create train/val/test splits."""
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

def main():
    with open("params.yaml", "r") as f:
        full_params = yaml.safe_load(f)

    print("Loaded params:", full_params)          # <-- debug line

    params = full_params["preprocess"]

    os.makedirs("data/processed", exist_ok=True)

    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test  = np.load("data/raw/x_test.npy")
    y_test  = np.load("data/raw/y_test.npy")

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Split validation set from training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train
    )

    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy",   x_val)
    np.save("data/processed/y_val.npy",   y_val)
    np.save("data/processed/x_test.npy",  x_test)
    np.save("data/processed/y_test.npy",  y_test)

    print("Processed data saved to data/processed/")
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

if __name__ == "__main__":
    main()