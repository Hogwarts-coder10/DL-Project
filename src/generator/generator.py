import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split

class LunarDataGenerator(Sequence):
    """Custom Data Generator for Lunar Terrain Segmentation."""

    def __init__(self, image_paths, mask_paths, batch_size=16, img_size=(256, 256), num_classes=4):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def color_to_class(self, rgb_mask):
        """
        Converts an RGB mask into a 2D array of class IDs.
        Adjust the RGB values if your dataset uses slightly different shades.
        """
        # Create an empty single-channel mask initialized with 0 (Background)
        class_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)

        # Define the RGB values for each class
        # (Assuming the mask was read and converted to RGB format)
        sky = [255, 0, 0]        # Red -> Class 1
        small_rocks = [0, 255, 0] # Green -> Class 2
        big_rocks = [0, 0, 255]  # Blue -> Class 3

        # Map the colors to their respective class indices
        class_mask[np.all(rgb_mask == sky, axis=-1)] = 1
        class_mask[np.all(rgb_mask == small_rocks, axis=-1)] = 2
        class_mask[np.all(rgb_mask == big_rocks, axis=-1)] = 3

        return class_mask

    def __getitem__(self, idx):
        """Generates and returns one batch of data."""
        # 1. Get the paths for the current batch
        batch_img_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        # 2. Initialize empty arrays for the batch
        x_batch = np.zeros((len(batch_img_paths), *self.img_size, 3), dtype=np.float32)
        y_batch = np.zeros((len(batch_mask_paths), *self.img_size, self.num_classes), dtype=np.float32)

        # 3. Load and preprocess the images and masks
        for i, (img_path, mask_path) in enumerate(zip(batch_img_paths, batch_mask_paths)):
            # --- IMAGE PROCESSING ---
            # Read and resize Image (Convert BGR to RGB)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            x_batch[i] = img / 255.0  # Normalize to [0, 1]

            # --- MASK PROCESSING ---
            # Read the mask as a color image (OpenCV reads as BGR, so convert to RGB)
            mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

            # Resize mask using nearest neighbor to prevent blending colors at edges
            mask_resized = cv2.resize(mask_rgb, self.img_size, interpolation=cv2.INTER_NEAREST)

            # Map the RGB colors to flat class IDs (0, 1, 2, 3)
            mask_class_ids = self.color_to_class(mask_resized)

            # One-hot encode the mask for the dice loss function
            mask_one_hot = to_categorical(mask_class_ids, num_classes=self.num_classes)
            y_batch[i] = mask_one_hot

        return x_batch, y_batch

def get_dataset_generators(image_dir, mask_dir, batch_size=16, img_size=(256, 256), num_classes=4, val_split=0.2):
    """Utility function to create training and validation generators."""

    # Grab all valid image and mask filenames
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    # Construct full paths
    image_paths = [os.path.join(image_dir, f) for f in image_filenames]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_filenames]

    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42
    )

    # Instantiate the custom generators
    train_gen = LunarDataGenerator(x_train, y_train, batch_size, img_size, num_classes)
    val_gen = LunarDataGenerator(x_val, y_val, batch_size, img_size, num_classes)

    print(f"--> Found {len(x_train)} training images and {len(x_val)} validation images.")
    return train_gen, val_gen
