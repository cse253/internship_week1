# Task 1: Capture video from webcam using OpenCV

import cv2

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0)

print("Webcam started. Press 'q' to quit.")

while cap.isOpened():
    # Read each frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Display the frame in a window
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
