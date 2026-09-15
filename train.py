import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# Import custom loss functions and data handlers from your refactored modules
from src.metrics.losses import dice_loss, dice_coef
from src.generator.generator import get_dataset_generators # Assuming your generator is here

def build_model_factory(model_name, input_shape=(256, 256, 3), num_classes=4):
    """Dynamically routes the model build based on the user's CLI choice."""
    if model_name == "unet":
        from src.models.unet import build_model
    elif model_name == "unet_plus_plus":
        from src.models.unetplusplus import build_model
    elif model_name == "attention_unet":
        from src.models.attention_unet import build_model
    else:
        raise ValueError(f"Model architecture '{model_name}' is not recognized.")
    
    print(f"--> Successfully loaded {model_name} architecture.")
    return build_model(input_shape=input_shape, num_classes=num_classes)

def main():
    # 1. Setup Argparse for CLI choices
    parser = argparse.ArgumentParser(description="Lunar Terrain Segmentation Training Pipeline")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["unet", "unet_plus_plus", "attention_unet"],
        help="Select the model architecture to train"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    
    args = parser.parse_args()

    # 2. Dynamic Routing for Checkpoints and Telemetry
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Variables dynamically adapt based on the chosen model
    model_save_path = f"saved_models/{args.model}_best.keras"
    csv_log_path = f"results/training_history_{args.model}.csv"

    print("\n" + "="*45)
    print("        TENSORFLOW TRAINING SESSION        ")
    print("="*45)
    print(f" Architecture : {args.model}")
    print(f" Epochs       : {args.epochs}")
    print(f" Batch Size   : {args.batch_size}")
    print(f" Checkpoint   : {model_save_path}")
    print("="*45 + "\n")

    # 3. Initialize Model
    model = build_model_factory(args.model)
    
    model.compile(
        optimizer="adam", 
        loss=dice_loss, 
        metrics=["accuracy", dice_coef]
    )

    # 4. Define Callbacks
    callbacks = [
        ModelCheckpoint(model_save_path, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1),
        CSVLogger(csv_log_path, append=False),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    # 5. Load Data and Train
    train_gen, val_gen = get_dataset_generators(
        image_dir = "lunar_dataset/images/render"
        mask_dir = "lunar_dataset/images/ground"
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
