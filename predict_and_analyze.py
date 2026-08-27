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

# Import custom metrics
from src.metrics import dice_loss, dice_coef

# --- Configuration Paths ---
# IMPORTANT: Update these paths to match your local directory structure
MODEL_PATH = "unet_plus_plus_best_10hr.keras"
CSV_PATH = "training_history_10hr.csv"
IMG_DIR = "lunar_dataset/images/render"
MASK_DIR = "lunar_dataset/images/ground"
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EVAL_SAMPLES = 500  # Number of samples to evaluate on local CPU


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

    # Loss
    axes[0].plot(epochs, df["loss"], label="Train Loss", color="#E64A19", lw=2)
    if "val_loss" in df.columns:
        axes[0].plot(epochs, df["val_loss"], label="Val Loss", color="#D32F2F", ls="--", lw=2)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, df["accuracy"], label="Train Accuracy", color="#1976D2", lw=2)
    if "val_accuracy" in df.columns:
        axes[1].plot(epochs, df["val_accuracy"], label="Val Accuracy", color="#0288D1", ls="--", lw=2)
    axes[1].set_title("Pixel Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Dice Score
    axes[2].plot(epochs, df["dice_coef"], label="Train Dice", color="#388E3C", lw=2)
    if "val_dice_coef" in df.columns:
        axes[2].plot(epochs, df["val_dice_coef"], label="Val Dice", color="#00796B", ls="--", lw=2)
    axes[2].set_title("Dice Coefficient")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Dice Score")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_curves_output.png", dpi=300)
    print("Saved 'training_curves_output.png'.")


def run_single_inference_and_feature_maps(model, sample_img_path):
    """Generates visual segmentation map and first Conv2D layer features."""
    if not os.path.exists(sample_img_path):
        print(f"Sample image '{sample_img_path}' not found.")
        return

    print(f"\nGenerating inference and feature maps for {sample_img_path}...")
    img = load_img(sample_img_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = img_to_array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # 1. Prediction (Handling multi-class output by taking argmax for display)
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
    plt.savefig("prediction_output.png", dpi=300)
    print("Saved 'prediction_output.png'.")

    # 2. Extract Feature Maps from first Conv2D layer
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
        plt.savefig("feature_maps_output.png", dpi=300)
        print("Saved 'feature_maps_output.png'.")


def evaluate_dataset(model, img_dir, mask_dir, num_samples=500):
    """Evaluates the dataset matching mask class mapping to model outputs."""
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
        # 1. Load and normalize RGB Input Image
        im = Image.open(img_paths[i]).convert("RGB").resize(target_size, Image.BILINEAR)
        im = np.array(im, dtype=np.float32) / 255.0
        
        # 2. Load Mask as integer class IDs (NEAREST neighbor interpolation)
        mk_raw = Image.open(mask_paths[i]).resize(target_size, Image.NEAREST)
        mk_arr = np.array(mk_raw)

        # Handle grayscale/palette class indices (0, 1, 2, 3...)
        if mk_arr.ndim == 3:
            mk_arr = mk_arr[:, :, 0]

        # Ensure class values are integer indices within [0, num_classes-1]
        if mk_arr.max() > num_classes - 1 and mk_arr.max() <= 255:
            unique_vals = np.unique(mk_arr)
            if len(unique_vals) <= num_classes:
                val_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                mk_arr = np.vectorize(val_map.get)(mk_arr)
            else:
                mk_arr = np.clip(mk_arr, 0, num_classes - 1)

        # 3. One-hot encode to target channel depth
        if num_classes > 1:
            mk_processed = to_categorical(mk_arr, num_classes=num_classes).astype(np.float32)
        else:
            mk_processed = (mk_arr > 0).astype(np.float32)[..., np.newaxis]

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


def main():
    # 1. Plot Curves
    plot_training_telemetry(CSV_PATH)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel file '{MODEL_PATH}' not found. Cannot proceed with evaluation.")
        return

    print(f"\nLoading weights from {MODEL_PATH}...")
    custom_objects = {"dice_loss": dice_loss, "dice_coef": dice_coef, "iou_metric": iou_metric}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    
    # Recompile with target metrics
    model.compile(optimizer="adam", loss=dice_loss, metrics=["accuracy", dice_coef, iou_metric])

    # 3. Single inference & feature maps
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")) + glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if all_imgs:
        run_single_inference_and_feature_maps(model, all_imgs[0])

    # 4. Quantitative Evaluation
    evaluate_dataset(model, IMG_DIR, MASK_DIR, num_samples=EVAL_SAMPLES)


if __name__ == "__main__":
    main()
