import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Step 1: Load and preprocess the data
data_path = "bowl2018"
desired_width = 256
desired_height = 256

def load_and_preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Preprocess the image (e.g., resizing, normalization)
    image = cv2.resize(image, (desired_width, desired_height))
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1] range

    return image

def combine_masks(masks):
    # Combine the instance masks into a single mask
    combined_mask = np.zeros_like(masks[0], dtype=np.float32)
    for mask in masks:
        combined_mask[mask > 0] = 1.0

    # Expand dimensions to make it (height, width, 1)
    return np.expand_dims(combined_mask, axis=-1)

def load_and_preprocess_mask(mask_paths):
    # Load masks using OpenCV
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths]

    # Preprocess the masks (e.g., resizing, normalization)
    masks = [cv2.resize(mask, (desired_width, desired_height)) for mask in masks]

    # Combine the masks into a single instance mask
    combined_mask = combine_masks(masks)

    return combined_mask

def create_train_val_datasets(data_path, test_size=0.2, batch_size=32):
    # Collect all image and mask directories
    image_dirs = [os.path.join(data_path, folder, "images") for folder in os.listdir(data_path)]
    mask_dirs = [os.path.join(data_path, folder, "masks") for folder in os.listdir(data_path)]

    # Load images and masks and preprocess them
    images = [load_and_preprocess_image(os.path.join(dir, os.listdir(dir)[0])) for dir in image_dirs]
    masks = [load_and_preprocess_mask([os.path.join(dir, mask_file) for mask_file in os.listdir(dir)]) for dir in mask_dirs]

    # Convert to numpy arrays
    images = np.array(images)
    masks = np.array(masks)

    # Step 2: Split the data into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=test_size, random_state=42)

    # Step 3: Create TensorFlow Datasets for train and validation
    def tf_dataset(images, masks, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        dataset = dataset.shuffle(buffer_size=len(images))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    # Create TensorFlow Datasets for train and validation
    train_dataset = tf_dataset(train_images, train_masks, batch_size=batch_size)
    val_dataset = tf_dataset(val_images, val_masks, batch_size=batch_size)

    return train_dataset, val_dataset


def create_test_dataset(data_path):
    # Collect all image directories for the test dataset
    image_dirs = [os.path.join(data_path, folder, "images") for folder in os.listdir(data_path)]

    # Load images and preprocess them
    images = [load_and_preprocess_image(os.path.join(dir, os.listdir(dir)[0])) for dir in image_dirs]

    # Convert to numpy array
    images = np.array(images)

    return images




def predict_and_plot(model, weight_name, test_images, num_samples_to_plot=4):
    # Define the DeeplabV3Plus model with the specified backbone and image size

    # Load the pre-trained weights
    model.load_weights(weight_name)

    # Make predictions on the test set
    predictions = model.predict(test_images)

    # Plot the first num_samples_to_plot test images along with their corresponding predictions
    plt.figure(figsize=(10, 10))

    for i in range(num_samples_to_plot):
        plt.subplot(4, num_samples_to_plot, i + 1)
        plt.imshow(test_images[i])
        plt.title(f"Image {i + 1}")
        plt.axis("off")

        plt.subplot(4, num_samples_to_plot, i + 1 + num_samples_to_plot)
        plt.imshow(predictions[i, ..., 0], cmap="gray")
        plt.title(f"Prediction {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


    
