import os
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import numpy as np
import random

# Directories for original and augmented images
sliced_images_dir = r'C:\Users\Peter\Downloads\Huron_data\Sliced_Images'
sliced_masks_dir = r'C:\Users\Peter\Downloads\Huron_data\Sliced_masks'
output_images_dir = r'C:\Users\Peter\Downloads\Huron_data\Aug_Images'
output_masks_dir = r'C:\Users\Peter\Downloads\Huron_data\Aug_masks'

# Create directories if they don't exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)


# Function to choose one of the fixed rotation angles
def random_rotation(image):
    angle = random.choice([0, 90, 180, 270])
    return transforms.functional.rotate(image, angle)


# Function to apply gamma correction
def gamma_correction(image, gamma=1.0):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(gamma)


# Function to apply gaussian noise
def add_gaussian_noise(image, mean=0, std=0.1):
    np_image = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, np_image.shape)
    np_image = np.clip(np_image + noise, 0, 1)
    noisy_image = (np_image * 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


# Applying the data transformations to the images
data_transforms = transforms.Compose([
    transforms.Lambda(lambda img: random_rotation(img)),
    transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.2)),  # Image scaling
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Brightness and contrast variability
    transforms.Lambda(lambda img: gamma_correction(img, gamma=random.uniform(0.8, 1.2))),  # Gamma correction
    transforms.Lambda(lambda img: add_gaussian_noise(img, mean=0, std=0.05))  # Gaussian noise
])

# Loading images and masks, augmenting them, and saving the augmented versions
images = os.listdir(sliced_images_dir)
masks = os.listdir(sliced_masks_dir)

for image, mask in zip(images[::3], masks[::3]):  # Iterate over every third image and mask
    # Load images and masks
    image_path = os.path.join(sliced_images_dir, image)
    mask_path = os.path.join(sliced_masks_dir, mask)

    image = Image.open(image_path).convert("RGB")  # Convert to RGB for compatibility with PIL
    mask = Image.open(mask_path)

    # Apply transformations and save augmented images and masks
    for i in range(5):  # Generate 5 augmented versions per image/mask
        # Apply the same transformation to both images and masks
        seed = np.random.randint(0, 10000)  # Same seed for reproducibility on both

        torch.manual_seed(seed)
        augmented_image = data_transforms(image)

        torch.manual_seed(seed)
        augmented_mask = data_transforms(mask)

        # Save augmented images and masks to the output directories
        aug_image_path = os.path.join(output_images_dir, f"aug_{i}_{image}")
        aug_mask_path = os.path.join(output_masks_dir, f"aug_{i}_{mask}")

        augmented_image.save(aug_image_path)
        augmented_mask.save(aug_mask_path)
