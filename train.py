import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

# ============================================================================
# GPU VRAM ALLOCATION SETUP
# ============================================================================
# Enable dynamic memory growth to prevent pre-allocation spikes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting error: {e}")
# ============================================================================

from src.metrics.losses import cce_loss, dice_coef
from src.generator.generator import get_dataset_generators


def build_model_factory(model_name, input_shape=(256, 256, 3), num_classes=4):
    """Dynamically routes the model build based on the user's CLI choice."""
    if model_name == "unet":
        from src.models.unet import build_model
    elif model_name == "unet_plus_plus":
        from src.models.unetplusplus import build_model
    elif model_name == "attention_unet":
        from src.models.attention_unet import build_model
    elif model_name == 'vnet':
        from src.models.vnet import build_model
    elif model_name == 'segnet':
        from src.models.segnet import build_model
    else:
        raise ValueError(f"Model architecture '{model_name}' is not recognized.")

    print(f"\n--> Successfully loaded {model_name} architecture.")
    return build_model(input_shape=input_shape, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser(description="Lunar Terrain Segmentation Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["unet", "unet_plus_plus", "attention_unet", "vnet", "segnet"],
        help="Select the model architecture to train"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")

    args = parser.parse_args()

    save_dir = '/content/drive/MyDrive/DL_Project/saved_models'
    log_dir = '/content/drive/MyDrive/DL_Project/results'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, f'{args.model}_best_10hr.keras')
    csv_log_path = os.path.join(log_dir, f'training_history_{args.model}_10hr.csv')

    print("\n" + "="*45)
    print("        TENSORFLOW TRAINING SESSION        ")
    print("="*45)
    print(f" Architecture : {args.model}")
    print(f" Epochs       : {args.epochs}")
    print(f" Batch Size   : {args.batch_size}")
    print(f" Checkpoint   : {model_save_path}")
    print("="*45 + "\n")

    model = build_model_factory(args.model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=cce_loss,
        metrics=["accuracy", dice_coef]
    )

    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coef',
            factor=0.5,
            patience=3,
            mode='max',
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coef',
            patience=8,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        CSVLogger(
            filename=csv_log_path,
            separator=',',
            append=False
        )
    ]

    train_gen, val_gen = get_dataset_generators(
        image_dir="lunar_dataset/images/render",
        mask_dir="lunar_dataset/images/ground",
        batch_size=args.batch_size
    )

    print("\nStarting training loop...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
