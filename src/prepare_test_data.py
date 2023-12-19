"""
Script to move a few images from the original dataset
into the test dataset folder randomly.
"""

import shutil
import glob
import random
import os
from tqdm import tqdm
from class_names import class_names as CLASS_NAMES  # Import the list of class names

random.seed(42)  # Set a seed for reproducibility

ROOT_DIR = os.path.join('..', 'plantvillage dataset', 'color')  # Root directory of the original dataset
DEST_DIR = os.path.join('..', 'input', 'test')  # Destination directory for the test dataset
# List of class directories from the imported class names
class_dirs = CLASS_NAMES
# Ratio of images to move to the test dataset
test_split = 0.2

# Loop through each class directory
for class_dir in class_dirs:
    os.makedirs(os.path.join(DEST_DIR, class_dir), exist_ok=True)  # Create destination directories for each class
    init_image_paths = glob.glob(os.path.join(ROOT_DIR, class_dir, "*"))  # Get initial image paths for the class
    print(f"Initial number of images for class {class_dir}: {len(init_image_paths)}")
    random.shuffle(init_image_paths)
    # Select a random subset of images for testing
    test_images = random.sample(init_image_paths, int(round(test_split * len(init_image_paths))))
    print(f"Copying {len(test_images)} images from {class_dir}")

    # Move selected test images to the test directory for the specific class
    for test_image_path in tqdm(test_images):
        image_name = test_image_path.split(os.path.sep)[-1]  # Extract the image name
        shutil.move(test_image_path, os.path.join(DEST_DIR, class_dir, image_name))  # Move the image to test directory
    final_image_paths = glob.glob(os.path.join(ROOT_DIR, class_dir, '*'))  # Get the final image paths after moving
    print(f"Final number of images for class {class_dir}: {len(final_image_paths)}\n")
