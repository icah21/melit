import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('../models/cacao_classifier.h5')

# Class labels (adjust according to your dataset)
CLASS_LABELS = ["Criollo", "Forastero", "Trinitario"]

# Load and preprocess the image for prediction
img_path = 'path_to_your_image.jpg'  # Update this path with the actual image you want to test
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))  # Resize to the model's input size
img = img / 255.0  # Normalize the image
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(img)
class_id = np.argmax(predictions)
label = CLASS_LABELS[class_id]
confidence = np.max(predictions) * 100

# Display the result
print(f"Predicted Class: {label} (Confidence: {confidence:.2f}%)")
