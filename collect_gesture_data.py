import cv2
import mediapipe as mp
import csv
import os

VALID_LABELS = ["positive", "negative", "neutral"]

print("Available emotion labels:", VALID_LABELS)
gesture_name = input("Enter emotion label to collect (positive / negative / neutral): ").strip().lower()

if gesture_name not in VALID_LABELS:
    print(f"Invalid label. Choose from: {VALID_LABELS}")
    exit(1)

os.makedirs("EmoSign/landmarks", exist_ok=True)
file_path = os.path.join("EmoSign", "landmarks", f"{gesture_name}.csv")

mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)
sample_count = 0
max_samples = 50
saving = False  # auto-save mode toggle

with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)

    header = ["label"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

    with mp_hands.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    h, w, _ = frame.shape
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Auto-save when saving mode is ON
                if saving:
                    for hand_landmarks in result.hand_landmarks:
                        row = [gesture_name]
                        for lm in hand_landmarks:
                            row.extend([lm.x, lm.y, lm.z])
                        writer.writerow(row)
                        sample_count += 1
                        print(f"Saved {sample_count}/{max_samples}")

                status = "SAVING..." if saving else "Press S to START saving"
                color = (0, 0, 255) if saving else (0, 255, 0)
                cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"{gesture_name}: {sample_count}/{max_samples}  Q=quit",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Collect Emotion Landmarks", frame)

            # waitKey(30) gives enough time to detect key press
            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                saving = not saving  # toggle save on/off
                print("Saving ON" if saving else "Saving OFF")

            if sample_count >= max_samples:
                print(f"Done! {max_samples} samples saved to {file_path}")
                break

            if key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
