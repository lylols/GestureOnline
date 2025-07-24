
with open('gestures.txt', 'r') as f:
    gestures = f.read().splitlines()

print("Available Gestures:")
for i, gesture in enumerate(gestures):
    print(f"{i}: {gesture}")

gesture_id = int(input("Enter gesture ID to record: "))
gesture_name = gestures[gesture_id]

import os
import csv

data_dir = os.path.join("gesture_data", gesture_name)
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, f"{gesture_name}.csv")

with open(csv_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = ['label']
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
    csv_writer.writerow(header)

print(f"[INFO] CSV file created for gesture: {gesture_name}")

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    sample_count = 0
    print("[INFO] Starting data collection. Press 'q' to quit.")

    while cap.isOpened() and sample_count < 300:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                row = [gesture_name]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                # Write to CSV
                with open(csv_path, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(row)
                    sample_count += 1
                    print(f"[INFO] Sample {sample_count} collected", end='\r')

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.namedWindow("Data Collection - Press q to exit", cv2.WINDOW_NORMAL)
        cv2.imshow("Data Collection - Press q to exit", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Collected {sample_count} samples for gesture '{gesture_name}'")
