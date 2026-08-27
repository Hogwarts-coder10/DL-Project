import numpy as np
np.long = np.int64   # type: ignore
np.ulong = np.uint64 # type: ignore

import os
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Import our custom modules
from src.metrics import dice_loss, dice_coef
from src.generator import LunarDataGenerator
from src.model import build_unet_plus_plus

def main():
    # 1. Setup File Paths
    image_dir = './lunar_dataset/images/render'
    mask_dir = './lunar_dataset/images/ground'
    
    all_images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    all_masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    assert len(all_images) == len(all_masks), "Mismatch between images and masks!"

    # 2. Split Data (80% Train, 20% Validation)
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=0.2, random_state=42
    )
    
    print(f"Training on {len(train_imgs)} images, validating on {len(val_imgs)} images.")

    # 3. Instantiate Memory-Safe Generators (4 classes)
    train_generator = LunarDataGenerator(train_imgs, train_masks, batch_size=8, num_classes=4, is_train=True)
    val_generator = LunarDataGenerator(val_imgs, val_masks, batch_size=8, num_classes=4, is_train=False)

    # 4. Build and Compile the Model (4 classes)
    model = build_unet_plus_plus(input_shape=(256, 256, 3), num_classes=4)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=['accuracy', dice_coef]
    )

    # 5. Setup Paths & Callbacks for 10-Hour Google Colab Run
    # Automatically saves directly to Google Drive if mounted, else falls back to local storage
    drive_dir = '/content/drive/MyDrive'
    save_dir = drive_dir if os.path.exists(drive_dir) else './saved_models'
    log_dir = drive_dir if os.path.exists(drive_dir) else './logs'
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        # Checkpoint: Saves best model to Drive to survive Colab session disconnections
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'unet_plus_plus_best_10hr.keras'),
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Reduce LR: Halves learning rate when validation dice plateaus for 3 epochs
        ReduceLROnPlateau(
            monitor='val_dice_coef',
            factor=0.5,
            patience=3,
            mode='max',
            min_lr=1e-7,
            verbose=1
        ),
        # Early Stopping: Halts training if no improvement after 6 epochs
        EarlyStopping(
            monitor='val_dice_coef',
            patience=6,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Telemetry: Saves training logs to CSV
        CSVLogger(
            filename=os.path.join(log_dir, 'training_history_10hr.csv'),
            separator=',',
            append=False
        )
    ]

    # 6. Execute Training Loop (Target: 58 Epochs / ~10 Hours)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=58,
        callbacks=callbacks,
        verbose="auto"
    )
    
    print(f"Training complete! Model and telemetry saved to: {save_dir}")

if __name__ == '__main__':
    main()