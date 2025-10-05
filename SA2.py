import cv2
import tensorflow as tf

# Step 1: Load your trained Teachable Machine model
model = tf.keras.models.load_model("keras_model.h5")

# Step 2: Load class labels (same order as in labels.txt)
CLASS_NAMES = [
    "Timothée Chalamet",
    "Zendaya",
    "Tom Cruise",
    "Angelina Jolie"
]

# Step 3: Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Step 4: Display live webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 5: Release resources
cap.release()
cv2.destroyAllWindows()
