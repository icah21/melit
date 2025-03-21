import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set up directories
data_dir = '../data/dataset_cropped_allclasses'  # Adjust path as needed
img_size = (224, 224)  # Resize images to this size

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

# Define CNN model
def create_model(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # For classification
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and compile the model
model = create_model()
model.summary()  # Print the model architecture

# Train the model
history = model.fit(
    train_gen,
    epochs=10,  # Set the number of epochs
    validation_data=val_gen
)

# Save the trained model
model.save('../models/cacao_classifier.h5')
