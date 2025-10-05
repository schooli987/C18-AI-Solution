import cv2
import numpy as np
import tensorflow as tf

# Load your trained Teachable Machine model
model = tf.keras.models.load_model("keras_model.h5")

# Labels must match your model's labels.txt
CLASS_NAMES = [
    "Timothée Chalamet",  
    "Zendaya",            # Hollywood
    "Tom Cruise",     # Hollywood
    "Angelina Jolie",      # Hollywood

]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess frame for model
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img.astype(np.float32)/255.0, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100
    label = f"{CLASS_NAMES[class_index]} ({confidence:.2f}%)"

    # Display label on webcam feed
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Celebrity Lookalike Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
