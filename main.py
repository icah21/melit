import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("cacao_classifier.h5")  # Ensure this model is trained

# Define class labels
CLASS_LABELS = ["Criollo", "Forastero", "Trinitario"]

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame")
        break

    # Preprocess image for model
    img = cv2.resize(frame, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    label = CLASS_LABELS[class_id]
    confidence = np.max(predictions) * 100

    # Display result
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Cacao Classification", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
