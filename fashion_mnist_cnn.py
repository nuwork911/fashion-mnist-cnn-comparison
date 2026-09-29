"""
Comparative analysis of neural network architectures on Fashion-MNIST.

Trains and evaluates three models:
  1. Baseline dense neural network (DNN)
  2. Unregularised CNN (shows overfitting)
  3. VGG-style CNN with dropout (final model)

Outputs (saved to results/): validation accuracy and loss curves,
confusion matrices and a per-class classification report.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, models

# Configuration
EPOCHS = 10
BATCH_SIZE = 64
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
tf.random.set_seed(42)
print("--- Environment set: seeds fixed ---")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# 1. Data loading and preprocessing
def load_data():
    print("\n[1] Loading Fashion-MNIST data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Class balance audit
    _, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution check: {dict(zip(CLASS_NAMES, counts))}")

    # Normalise to the 0-1 range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(f"Training set: {len(x_train)} | Test set: {len(x_test)}")
    return x_train, y_train, x_test, y_test


# 2. Model definitions
def create_baseline_dnn():
    """Fully connected network that ignores spatial structure."""
    model = models.Sequential([
        layers.Dense(512, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="Baseline_DNN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def create_overfit_cnn():
    """CNN without dropout, used to demonstrate overfitting."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="Unregularised_CNN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def create_final_cnn():
    """VGG-style CNN with stacked convolutions and dropout."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="Final_CNN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# 3. Plotting helpers
def plot_metrics(dnn, no_drop, final):
    for metric, title in [("val_accuracy", "Validation Accuracy"),
                          ("val_loss", "Validation Loss")]:
        plt.figure(figsize=(10, 6))
        plt.plot(dnn.history[metric], label="Baseline DNN", linestyle=":", color="orange")
        plt.plot(no_drop.history[metric], label="CNN (no dropout)", linestyle="--", color="red")
        plt.plot(final.history[metric], label="Final CNN", linewidth=2.5, color="green")
        plt.title(f"{title} Comparison")
        plt.xlabel("Epochs")
        plt.ylabel(title.split()[-1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}.png"), bbox_inches="tight")
        plt.close()


def plot_confusion(y_true, pred_dnn, pred_cnn):
    plt.figure(figsize=(16, 7))
    for i, (preds, title, cmap) in enumerate([
            (pred_dnn, "Baseline DNN Confusion Matrix", "Blues"),
            (pred_cnn, "Final CNN Confusion Matrix", "Greens")]):
        plt.subplot(1, 2, i + 1)
        sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt="d", cmap=cmap,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(title)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices_run.png"), bbox_inches="tight")
    plt.close()


def main():
    x_train, y_train, x_test, y_test = load_data()

    # Flattened input for the DNN, (28, 28, 1) tensors for the CNNs
    x_train_flat, x_test_flat = x_train.reshape((-1, 784)), x_test.reshape((-1, 784))
    x_train_cnn, x_test_cnn = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)

    fit_args = dict(epochs=EPOCHS, validation_split=0.2, batch_size=BATCH_SIZE, verbose=1)

    print("\n[2] Training baseline DNN...")
    dnn_model = create_baseline_dnn()
    dnn_history = dnn_model.fit(x_train_flat, y_train, **fit_args)

    print("\n[3] Training unregularised CNN...")
    cnn_no_drop = create_overfit_cnn()
    no_drop_history = cnn_no_drop.fit(x_train_cnn, y_train, **fit_args)

    print("\n[4] Training final CNN...")
    cnn_model = create_final_cnn()
    cnn_history = cnn_model.fit(x_train_cnn, y_train, **fit_args)

    print("\n[5] Evaluating and saving results...")
    for name, model, x in [("Baseline DNN", dnn_model, x_test_flat),
                           ("Unregularised CNN", cnn_no_drop, x_test_cnn),
                           ("Final CNN", cnn_model, x_test_cnn)]:
        _, acc = model.evaluate(x, y_test, verbose=0)
        print(f"{name}: test accuracy = {acc:.4f}")

    plot_metrics(dnn_history, no_drop_history, cnn_history)

    y_pred_dnn = np.argmax(dnn_model.predict(x_test_flat), axis=1)
    y_pred_cnn = np.argmax(cnn_model.predict(x_test_cnn), axis=1)

    report = classification_report(y_test, y_pred_cnn, target_names=CLASS_NAMES,
                                   output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(3)
    print("\nFinal CNN classification report:")
    print(df_report)
    df_report.to_csv(os.path.join(RESULTS_DIR, "cnn_classification_report.csv"))

    plot_confusion(y_test, y_pred_dnn, y_pred_cnn)
    print(f"\nDone. Results saved to '{RESULTS_DIR}/'.")


if __name__ == "__main__":
    main()
