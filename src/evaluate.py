"""Evaluate the trained model and write metrics."""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def main():
    model = tf.keras.models.load_model("models/model.h5")

    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    # Confusion matrix image
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – Fashion-MNIST")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
        )
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("metrics.json and confusion_matrix.png written.")

if __name__ == "__main__":
    main()