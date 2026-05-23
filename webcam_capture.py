# Task 1: Capture video from webcam using OpenCV
# Integrated with ASL Alphabet labels

import cv2
import pickle
import os

# Load trained ASL Alphabet model if available
model = None
if os.path.exists("asl_model.pkl"):
    with open("asl_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("ASL Alphabet model loaded.")
else:
    print("No trained ASL model found. Run train_model.py first.")

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Show model status on screen
    status = "ASL Alphabet Model: Ready" if model else "ASL Alphabet Model: Not Loaded"
    color = (0, 255, 0) if model else (0, 0, 255)
    cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Webcam Feed - ASL Alphabet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
