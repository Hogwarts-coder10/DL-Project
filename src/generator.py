import cv2
import numpy as np
import random
import tensorflow as tf
from keras.utils import Sequence, to_categorical

class LunarDataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=16, img_size=(256, 256), num_classes=3, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.is_train = is_train
        
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Fetch file paths for the current batch
        batch_img_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Initialize batch arrays
        batch_x = np.zeros((len(batch_img_paths), *self.img_size, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_mask_paths), *self.img_size, self.num_classes), dtype=np.float32)

        for i, (img_path, mask_path) in enumerate(zip(batch_img_paths, batch_mask_paths)):
            # 1. Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image at: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            # 2. Load and resize mask (Nearest Neighbor to preserve class labels)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load image at: {mask_path}")

            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            # 3. Apply matching augmentations to BOTH if training
            if self.is_train:
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)
                
                if random.random() > 0.5:
                    img = cv2.flip(img, 0)
                    mask = cv2.flip(mask, 0)
                    
                k = random.randint(0, 3)
                if k > 0:
                    img = np.rot90(img, k)
                    mask = np.rot90(mask, k)

            # 4. Normalize image and one-hot encode mask
            batch_x[i] = img / 255.0  
            batch_y[i] = to_categorical(mask, num_classes=self.num_classes)

        return batch_x, batch_y
