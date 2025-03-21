import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, filedialog

# ✅ Set the phone camera URL
PHONE_CAMERA_URL = "http://192.168.254.140:8080/video"  # Change this to your phone's IP

# ✅ Initialize Tkinter Dashboard
root = tk.Tk()
root.title("Cacao Bean Classification Dashboard")
root.geometry("400x200")

# ✅ Label for detected bean type
bean_label = Label(root, text="Detected Bean: None", font=("Arial", 16))
bean_label.pack(pady=10)

# ✅ Dictionary to store uploaded bean reference images
bean_images = {}

def upload_image(bean_type):
    """Uploads an image and extracts HSV color range for classification."""
    file_path = filedialog.askopenfilename(title=f"Upload {bean_type} Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        bean_images[bean_type] = extract_hsv_range(image)
        print(f"✅ {bean_type} image uploaded! Extracted HSV range: {bean_images[bean_type]}")

def extract_hsv_range(image):
    """Extracts HSV color range from an image for classification."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_image.reshape(-1, 3), axis=0)
    lower = np.clip(mean_hsv - 20, 0, 255).astype("uint8")
    upper = np.clip(mean_hsv + 20, 0, 255).astype("uint8")
    return (tuple(lower), tuple(upper))

# ✅ Upload images for Criollo, Forastero, and Trinitario
upload_image("Criollo")
upload_image("Forastero")
upload_image("Trinitario")

# ✅ Open the phone camera stream
cap = cv2.VideoCapture(PHONE_CAMERA_URL)
if not cap.isOpened():
    print("❌ Error: Could not open phone camera stream.")
    exit()
else:
    print("✅ Connected to Phone Camera!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't read frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_beans = []

    for cacao_type, hsv_range in bean_images.items():
        lower_bound = np.array(hsv_range[0], dtype="uint8")
        upper_bound = np.array(hsv_range[1], dtype="uint8")

        # ✅ Create binary mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.medianBlur(mask, 7)  # Reduce noise

        # ✅ Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # ✅ Ignore very small objects
            if area < 800:
                continue

            # ✅ Bean-like shape detection: Approximate an ellipse
            ellipse = cv2.fitEllipse(contour) if len(contour) >= 5 else None
            if ellipse:
                x, y, w, h = cv2.boundingRect(contour)

                # ✅ Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # ✅ Add detected bean type to list
                detected_beans.append(cacao_type)

    # ✅ Update the Tkinter dashboard with the detected type
    detected_text = " | ".join(set(detected_beans)) if detected_beans else "None"
    bean_label.config(text=f"Detected Bean: {detected_text}")

    # ✅ Show the output
    cv2.imshow('Cacao Bean Classification', frame)
    root.update_idletasks()
    root.update()

    # ✅ Exit when 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
root.destroy()
