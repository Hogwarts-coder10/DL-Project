import numpy as np
np.long = np.int64   # type: ignore
np.ulong = np.uint64 # type: ignore

import os
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
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

    # 3. Instantiate Memory-Safe Generators (Updated for 4 classes)
    train_generator = LunarDataGenerator(train_imgs, train_masks, batch_size=8, num_classes=4, is_train=True)
    val_generator = LunarDataGenerator(val_imgs, val_masks, batch_size=8, num_classes=4, is_train=False)

    # 4. Build and COMPILE the Model (Updated for 4 classes)
    model = build_unet_plus_plus(input_shape=(256, 256, 3), num_classes=4)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=['accuracy', dice_coef]
    )

    # 5. Setup Callbacks
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath='saved_models/unet_plus_plus_best.keras',
        monitor='val_dice_coef',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    csv_logger = CSVLogger('logs/training_history.csv')

    # 6. Execute Training Loop
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=1,
        callbacks=[checkpoint, csv_logger],
        verbose="auto"
    )
    
    print("Training complete! Best model saved to 'saved_models/'")

if __name__ == '__main__':
    main()
