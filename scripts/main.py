import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the data directory
data_dir = 'data/dataset_cropped_allclasses'  # Change to your actual path
img_size = (224, 224)  # Resize all images to a consistent size

# Create a data generator for loading and augmenting images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load the data
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
