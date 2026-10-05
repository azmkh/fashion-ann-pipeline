"""Build, train and save the ANN."""
import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    os.makedirs("models", exist_ok=True)

    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val   = np.load("data/processed/x_val.npy")
    y_val   = np.load("data/processed/y_val.npy")

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(params["dense_units"], activation="relu"),
        Dropout(params["dropout_rate"]),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2
    )

    model.save("models/model.h5")
    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)

    print("Model saved to models/model.h5")
    print("History saved to models/history.csv")

if __name__ == "__main__":
    main()