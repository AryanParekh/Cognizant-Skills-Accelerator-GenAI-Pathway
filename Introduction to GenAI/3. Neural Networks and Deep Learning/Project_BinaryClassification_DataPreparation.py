import os
import numpy as np
from PIL import Image

seed = 42
np.random.seed(seed)

# Part 1.1: Choose a dataset for binary classification and get data ready
train_dir = 'PetImages'
categories = ['Cat', 'Dog']

file_paths = []
labels = []

def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except Exception:
        return True

for category in categories:
    category_dir = os.path.join(train_dir, category)
    for file_name in os.listdir(category_dir):
        file_path = os.path.join(category_dir, file_name)
        if not is_image_corrupted(file_path):
            file_paths.append(file_path)
            labels.append(category)

file_paths = np.array(file_paths)
labels = np.array(labels)

np.savez('valid_data.npz', file_paths=file_paths, labels=labels)