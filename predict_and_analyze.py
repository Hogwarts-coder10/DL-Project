import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model

# Import custom metrics to load the model successfully
from src.metrics import dice_loss, dice_coef

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Loads an image, resizes it, and scales pixels to [0, 1]."""
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img, img_input

def plot_training_history(csv_path):
    """Plots loss, accuracy, and dice coefficient curves from the CSV log."""
    if not os.path.exists(csv_path):
        print(f"Warning: Training history file not found at {csv_path}")
        return
        
    history = pd.read_csv(csv_path)
    epochs = history['epoch'] + 1 # 1-indexed for plotting
    
    plt.figure(figsize=(18, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Val Loss', color='#c0392b', linestyle='--', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 2. Accuracy Curve
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy', color='#3498db', linewidth=2)
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], label='Val Accuracy', color='#2980b9', linestyle='--', linewidth=2)
    plt.title('Pixel Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 3. Dice Coefficient Curve
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['dice_coef'], label='Train Dice', color='#2ecc71', linewidth=2)
    if 'val_dice_coef' in history:
        plt.plot(epochs, history['val_dice_coef'], label='Val Dice', color='#27ae60', linestyle='--', linewidth=2)
    plt.title('Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_curves_output.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_feature_maps(model, img_tensor, max_filters=16):
    """Extracts and plots feature maps from the first Conv2D layer to interpret learned features."""
    # Dynamically locate the first convolutional layer
    target_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break
            
    if not target_layer:
        print("No Conv2D layer found for feature map extraction.")
        return

    print(f"Extracting feature maps from layer: {target_layer.name}")
    
    # Create a sub-model that outputs the intermediate activations
    feature_extractor = Model(inputs=model.inputs, outputs=target_layer.output)
    feature_maps = feature_extractor.predict(img_tensor, verbose=0)
    
    # Plot a grid of the filter activations (max 16 to keep it readable)
    num_filters = feature_maps.shape[-1]
    filters_to_plot = min(num_filters, max_filters)
    grid_size = int(np.ceil(np.sqrt(filters_to_plot)))
    
    plt.figure(figsize=(10, 10))
    for i in range(filters_to_plot):
        plt.subplot(grid_size, grid_size, i + 1)
        # Using a distinct colormap for feature activations
        plt.imshow(feature_maps[0, :, :, i], cmap='magma')
        plt.axis('off')
        plt.title(f'Filter {i+1}', fontsize=10)
        
    plt.suptitle(f'Interpretable Feature Maps ({target_layer.name})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('feature_maps_output.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_prediction(original_img, predicted_mask):
    """Plots the original image next to the model's predicted mask."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Input Lunar Terrain")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmentation Mask")
    plt.imshow(predicted_mask, cmap='viridis') 
    plt.axis('off')
    
    plt.savefig('prediction_output.png', bbox_inches='tight')
    plt.show()

def main():
    # 1. Define Paths (Update these based on your local or Drive setup)
    model_path = '/content/drive/MyDrive/unet_plus_plus_best_10hr.keras'
    csv_path = '/content/drive/MyDrive/training_history_10hr.csv'
    test_image_path = './lunar_dataset/images/render/test_image_001.png'
    
    # 2. Plot Training Curves from CSV
    print("Generating training curves...")
    plot_training_history(csv_path)

    if not os.path.exists(model_path) or not os.path.exists(test_image_path):
        print("Model or test image not found. Ensure paths are correct to run inference.")
        return

    # 3. Load the Model
    print("Loading model for inference...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef}
    )
    
    # 4. Preprocess Image
    raw_img, img_tensor = load_and_preprocess_image(test_image_path)
    
    # 5. Extract and Plot Feature Maps
    print("Visualizing internal representations...")
    visualize_feature_maps(model, img_tensor)
    
    # 6. Run Final Segmentation Inference
    print("Generating final prediction mask...")
    prediction = model.predict(img_tensor, verbose=0)
    predicted_mask = np.argmax(prediction[0], axis=-1)
    
    # 7. Plot Prediction
    visualize_prediction(raw_img, predicted_mask)

if __name__ == '__main__':
    main()