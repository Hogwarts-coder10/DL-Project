import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import custom metrics from your newly refactored module structure
from src.metrics.losses import dice_loss, dice_coef

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "unet_plus_plus"
MODEL_PATH = "unet_plus_plus_best_10hr.keras"
CSV_PATH = "training_history_10hr.csv"

IMG_DIR = "lunar_dataset/images/render"
MASK_DIR = "lunar_dataset/images/ground"
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EVAL_SAMPLES = 500

CLASS_NAMES = ["Background", "Sky", "Small Rock", "Big Rock"]
# ============================================================================

def iou_metric(y_true, y_pred, smooth=1e-6):
    """Computes Intersection over Union (Jaccard Index)."""
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred_bin, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def plot_training_telemetry(csv_file):
    """Plots training vs validation loss, accuracy, and Dice scores."""
    if not os.path.exists(csv_file):
        print(f"Telemetry log '{csv_file}' not found. Skipping curve plotting.")
        return

    print("\nGenerating training telemetry plots...")
    df = pd.read_csv(csv_file)
    epochs = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, df["loss"], label="Train Loss", color="#E64A19", lw=2)
    if "val_loss" in df.columns:
        axes[0].plot(epochs, df["val_loss"], label="Val Loss", color="#D32F2F", ls="--", lw=2)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, df["accuracy"], label="Train Accuracy", color="#1976D2", lw=2)
    if "val_accuracy" in df.columns:
        axes[1].plot(epochs, df["val_accuracy"], label="Val Accuracy", color="#0288D1", ls="--", lw=2)
    axes[1].set_title("Pixel Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    axes[2].plot(epochs, df["dice_coef"], label="Train Dice", color="#388E3C", lw=2)
    if "val_dice_coef" in df.columns:
        axes[2].plot(epochs, df["val_dice_coef"], label="Val Dice", color="#00796B", ls="--", lw=2)
    axes[2].set_title("Dice Coefficient")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Dice Score")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    out_path = f"results/training_curves_{MODEL_NAME}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved '{out_path}'.")


def generate_comparison_grid(model, img_paths, mask_paths, num_samples=5):
    """Generates a multi-row grid comparing Input, Ground Truth, and Predictions."""
    print(f"\nGenerating {num_samples}-image comparison grid...")

    # Grab a random sample of matched images/masks
    indices = random.sample(range(len(img_paths)), min(num_samples, len(img_paths)))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    fig.suptitle(f"Segmentation Results - {MODEL_NAME}", fontsize=16, y=0.98)

    for i, idx in enumerate(indices):
        # 1. Load Input Image
        im_pil = Image.open(img_paths[idx]).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        im_arr = np.array(im_pil, dtype=np.float32) / 255.0

        # 2. Load Ground Truth (Using the exact RGB Mapping logic)
        mk_raw = Image.open(mask_paths[idx]).convert("RGB").resize(IMG_SIZE, Image.NEAREST)
        mk_arr = np.array(mk_raw)

        gt_mask = np.zeros(IMG_SIZE, dtype=np.uint8)
        gt_mask[np.all(mk_arr == [255, 0, 0], axis=-1)] = 1  # Sky
        gt_mask[np.all(mk_arr == [0, 255, 0], axis=-1)] = 2  # Small Rock
        gt_mask[np.all(mk_arr == [0, 0, 255], axis=-1)] = 3  # Big Rock

        # 3. Model Prediction
        input_tensor = np.expand_dims(im_arr, axis=0)
        pred_probs = model.predict(input_tensor, verbose=0)[0]
        pred_mask = np.argmax(pred_probs, axis=-1)

        # Plotting
        axes[i, 0].imshow(im_arr)
        axes[i, 0].axis("off")
        if i == 0: axes[i, 0].set_title("Input Image", fontsize=14)

        axes[i, 1].imshow(gt_mask, cmap="viridis", vmin=0, vmax=3)
        axes[i, 1].axis("off")
        if i == 0: axes[i, 1].set_title("Ground Truth Mask", fontsize=14)

        axes[i, 2].imshow(pred_mask, cmap="viridis", vmin=0, vmax=3)
        axes[i, 2].axis("off")
        if i == 0: axes[i, 2].set_title("Predicted Mask", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = f"results/comparison_grid_{MODEL_NAME}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved '{out_path}'.")


def run_feature_map_extraction(model, sample_img_path):
    """Extracts interpretable Conv2D features for the rubric requirement."""
    print(f"\nExtracting feature maps for {sample_img_path}...")
    img = load_img(sample_img_path, target_size=IMG_SIZE, color_mode="rgb")
    input_tensor = np.expand_dims(img_to_array(img) / 255.0, axis=0)

    first_conv_layer = None
    for layer in model.layers:
        if "conv2d" in layer.name.lower():
            first_conv_layer = layer
            break

    if first_conv_layer:
        activation_model = Model(inputs=model.inputs, outputs=first_conv_layer.output)
        activations = activation_model.predict(input_tensor, verbose=0)
        num_filters = min(16, activations.shape[-1])

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f"Interpretable Feature Maps ({first_conv_layer.name})", fontsize=14)
        for i in range(num_filters):
            row, col = divmod(i, 4)
            axes[row, col].imshow(activations[0, :, :, i], cmap="magma")
            axes[row, col].set_title(f"Filter {i+1}", fontsize=10)
            axes[row, col].axis("off")
        plt.tight_layout()
        out_path = f"results/feature_maps_{MODEL_NAME}.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved '{out_path}'.")


def evaluate_dataset(model, img_dir, mask_dir, num_samples=500):
    """Evaluates the dataset and generates a confusion matrix."""
    target_h, target_w = IMG_SIZE
    num_classes = model.output_shape[-1] if model.output_shape[-1] is not None else 1

    print(f"\nModel Configuration -> Resolution: {(target_h, target_w)}, Classes: {num_classes}")

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    img_paths, mask_paths = [], []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        mask_paths.extend(glob.glob(os.path.join(mask_dir, ext)))

    img_paths.sort()
    mask_paths.sort()
    total_matched = min(len(img_paths), len(mask_paths))

    if total_matched == 0:
        print("No matched image-mask pairs found for evaluation.")
        return

    sample_count = min(num_samples, total_matched)
    print(f"Running evaluation across {sample_count} samples...")

    batch_imgs, batch_masks = [], []

    for i in range(sample_count):
        im = Image.open(img_paths[i]).convert("RGB").resize((target_w, target_h), Image.BILINEAR)
        batch_imgs.append(np.array(im, dtype=np.float32) / 255.0)

        mk_raw = Image.open(mask_paths[i]).convert("RGB").resize((target_w, target_h), Image.NEAREST)
        mk_arr = np.array(mk_raw)

        index_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        index_mask[np.all(mk_arr == [255, 0, 0], axis=-1)] = 1
        index_mask[np.all(mk_arr == [0, 255, 0], axis=-1)] = 2
        index_mask[np.all(mk_arr == [0, 0, 255], axis=-1)] = 3

        batch_masks.append(to_categorical(index_mask, num_classes=num_classes).astype(np.float32))

    x_eval, y_eval = np.array(batch_imgs), np.array(batch_masks)
    results = model.evaluate(x_eval, y_eval, batch_size=BATCH_SIZE, verbose=1)

    print("\n" + "=" * 45)
    print("        QUANTITATIVE EVALUATION REPORT       ")
    print("=" * 45)
    print(f" Test Loss (Dice Loss):  {results[0]:.4f}")
    print(f" Pixel Accuracy:         {results[1] * 100:.2f}%")
    print(f" Mean Dice Coefficient:  {results[2]:.4f}")
    if len(results) > 3: print(f" Mean IoU (Jaccard):     {results[3]:.4f}")
    print("=" * 45)

    print("\nGenerating confusion matrix...")
    y_pred_probs = model.predict(x_eval, batch_size=BATCH_SIZE, verbose=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=-1).flatten()
    y_true_labels = np.argmax(y_eval, axis=-1).flatten()

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(range(num_classes)))
    cm_normalized = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (counts) - {MODEL_NAME}")

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title(f"Confusion Matrix (row-normalized) - {MODEL_NAME}")

    plt.tight_layout()
    out_path = f"results/confusion_matrix_{MODEL_NAME}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved '{out_path}'.")


def main():
    # 1. Telemetry
    plot_training_telemetry(CSV_PATH)

    # 2. Model Loading
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel file '{MODEL_PATH}' not found.")
        return

    print(f"\nLoading weights from {MODEL_PATH}...")
    custom_objects = {"dice_loss": dice_loss, "dice_coef": dice_coef, "iou_metric": iou_metric}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    model.compile(optimizer="adam", loss=dice_loss, metrics=["accuracy", dice_coef, iou_metric])

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    img_paths, mask_paths = [], []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(IMG_DIR, ext)))
        mask_paths.extend(glob.glob(os.path.join(MASK_DIR, ext)))
    img_paths.sort()
    mask_paths.sort()

    if img_paths and mask_paths:
        # 3. Generate the 5-Image Grid
        generate_comparison_grid(model, img_paths, mask_paths, num_samples=5)

        # 4. Extract Interpretability Feature Maps (for one random image)
        run_feature_map_extraction(model, random.choice(img_paths))

    # 5. Full Evaluation
    evaluate_dataset(model, IMG_DIR, MASK_DIR, num_samples=EVAL_SAMPLES)


if __name__ == "__main__":
    main()
