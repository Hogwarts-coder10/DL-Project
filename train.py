import os
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Import our custom modules
from src.metrics import dice_loss, dice_coef
from src.generator import LunarDataGenerator
from src.model import build_unet_plus_plus

def main():
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

    # 3. Instantiate Memory-Safe Generators
    train_generator = LunarDataGenerator(train_imgs, train_masks, batch_size=16, is_train=True)
    val_generator = LunarDataGenerator(val_imgs, val_masks, batch_size=16, is_train=False)

    # 4. Build and COMPILE the Model
    model = build_unet_plus_plus(input_shape=(256, 256, 3), num_classes=3)
    
    # Here is the compile step you were looking for!
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=['accuracy', dice_coef]
    )

    # 5. Setup Callbacks (Save the best model automatically)
    os.makedirs('saved_models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath='saved_models/unet_plus_plus_best.keras',
        monitor='val_dice_coef',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # 6. Execute Training Loop
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30, 
        callbacks=[checkpoint],
        verbose='auto'
    )
    
    print("Training complete! Best model saved to 'saved_models/'")

if __name__ == '__main__':
    main()
