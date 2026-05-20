import cv2
import mediapipe as mp
import csv
import os

gesture_name = input("Enter gesture name: ")

os.makedirs("dataset", exist_ok=True)
file_path = f"dataset/{gesture_name}.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sample_count = 0
max_samples = 30

with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)

    header = ["gesture"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

    with mp_hands.Hands(max_num_hands=1) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(
                frame,
                f"{gesture_name}: {sample_count}/30",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Gesture Collection", frame)

            key = cv2.waitKey(1)

            if key == ord('s') and result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    row = [gesture_name]

                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])

                    writer.writerow(row)
                    sample_count += 1
                    print("Saved", sample_count)

            if sample_count >= 30:
                break

            if key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()