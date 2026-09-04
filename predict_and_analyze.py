import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import custom metrics
from src.metrics import dice_loss, dice_coef

# ============================================================================
# CONFIGURATION -- change these 3 lines to switch between models
# ============================================================================
MODEL_NAME = "unet_plus_plus"          # e.g. "unet_plus_plus" or "unet"
MODEL_PATH = "unet_plus_plus_best_10hr.keras"   # or "saved_models/unet_best.keras"
CSV_PATH = "training_history_10hr.csv"          # or "logs/training_history_unet.csv"
# ============================================================================

IMG_DIR = "lunar_dataset/images/render"
MASK_DIR = "lunar_dataset/images/ground"
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EVAL_SAMPLES = 500  # Number of samples to evaluate on local CPU

CLASS_NAMES = ["Background", "Sky", "Small Rock", "Big Rock"]


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
    out_path = f"training_curves_{MODEL_NAME}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved '{out_path}'.")


def run_single_inference_and_feature_maps(model, sample_img_path):
    """Generates visual segmentation map and first Conv2D layer features."""
    if not os.path.exists(sample_img_path):
        print(f"Sample image '{sample_img_path}' not found.")
        return

    print(f"\nGenerating inference and feature maps for {sample_img_path}...")
    img = load_img(sample_img_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = img_to_array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    pred_mask = model.predict(input_tensor, verbose=0)[0]
    num_classes = pred_mask.shape[-1]

    if num_classes > 1:
        pred_display = np.argmax(pred_mask, axis=-1)
    else:
        pred_display = (pred_mask > 0.5).astype(np.float32).squeeze()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_array)
    ax[0].set_title("Input Lunar Terrain")
    ax[0].axis("off")

    ax[1].imshow(pred_display, cmap="viridis")
    ax[1].set_title("Predicted Segmentation Mask")
    ax[1].axis("off")
    plt.tight_layout()
    out_path = f"prediction_{MODEL_NAME}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved '{out_path}'.")

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
        out_path = f"feature_maps_{MODEL_NAME}.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved '{out_path}'.")


def evaluate_dataset(model, img_dir, mask_dir, num_samples=500):
    """Evaluates the dataset matching mask class mapping to model outputs,
    and generates a confusion matrix over per-pixel class predictions."""
    target_h = model.input_shape[1] if model.input_shape[1] is not None else 256
    target_w = model.input_shape[2] if model.input_shape[2] is not None else 256
    target_size = (target_w, target_h)
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
    print(f"Running evaluation across {sample_count} samples with class encoding...")

    batch_imgs, batch_masks = [], []

    for i in range(sample_count):
        im = Image.open(img_paths[i]).convert("RGB").resize(target_size, Image.BILINEAR)
        im = np.array(im, dtype=np.float32) / 255.0

        mk_raw = Image.open(mask_paths[i]).convert("RGB").resize(target_size, Image.NEAREST)
        mk_arr = np.array(mk_raw)

        # Map exact RGB colors to class indices, matching src/generator.py exactly:
        # 0 = Background, 1 = Sky (red), 2 = Small Rock (green), 3 = Big Rock (blue)
        index_mask = np.zeros(target_size[::-1], dtype=np.uint8)
        index_mask[np.all(mk_arr == [255, 0, 0], axis=-1)] = 1
        index_mask[np.all(mk_arr == [0, 255, 0], axis=-1)] = 2
        index_mask[np.all(mk_arr == [0, 0, 255], axis=-1)] = 3

        if num_classes > 1:
            mk_processed = to_categorical(index_mask, num_classes=num_classes).astype(np.float32)
        else:
            mk_processed = (index_mask > 0).astype(np.float32)[..., np.newaxis]

        batch_imgs.append(im)
        batch_masks.append(mk_processed)

    x_eval = np.array(batch_imgs, dtype=np.float32)
    y_eval = np.array(batch_masks, dtype=np.float32)

    results = model.evaluate(x_eval, y_eval, batch_size=BATCH_SIZE, verbose=1)

    print("\n" + "=" * 45)
    print("        QUANTITATIVE EVALUATION REPORT       ")
    print("=" * 45)
    print(f" Test Loss (Dice Loss):  {results[0]:.4f}")
    print(f" Pixel Accuracy:         {results[1] * 100:.2f}%")
    print(f" Mean Dice Coefficient:  {results[2]:.4f}")
    if len(results) > 3:
        print(f" Mean IoU (Jaccard):     {results[3]:.4f}")
    print("=" * 45)

    # --- Confusion Matrix (per-pixel, across all evaluated samples) ---
    print("\nGenerating confusion matrix...")
    y_pred_probs = model.predict(x_eval, batch_size=BATCH_SIZE, verbose=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=-1).flatten()
    y_true_labels = np.argmax(y_eval, axis=-1).flatten()

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    cm_normalized = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    display_names = CLASS_NAMES[:num_classes] if num_classes <= len(CLASS_NAMES) else [str(i) for i in labels]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=display_names, yticklabels=display_names, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (counts) - {MODEL_NAME}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=display_names, yticklabels=display_names, ax=axes[1])
    axes[1].set_title(f"Confusion Matrix (row-normalized) - {MODEL_NAME}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    out_path = f"confusion_matrix_{MODEL_NAME}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved '{out_path}'.")


def main():
    plot_training_telemetry(CSV_PATH)

    if not os.path.exists(MODEL_PATH):
        print(f"\nModel file '{MODEL_PATH}' not found. Cannot proceed with evaluation.")
        return

    print(f"\nLoading weights from {MODEL_PATH}...")
    custom_objects = {"dice_loss": dice_loss, "dice_coef": dice_coef, "iou_metric": iou_metric}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

    model.compile(optimizer="adam", loss=dice_loss, metrics=["accuracy", dice_coef, iou_metric])

    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")) + glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if all_imgs:
        run_single_inference_and_feature_maps(model, all_imgs[0])

    evaluate_dataset(model, IMG_DIR, MASK_DIR, num_samples=EVAL_SAMPLES)


if __name__ == "__main__":
    main()