import cv2
import mediapipe as mp
import joblib
import numpy as np
import ctypes
import time

model = joblib.load('gesture_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pause_count = 0
not_understood_count = 0
not_audible_count = 0
threshold = 60

reset_interval = 10
last_reset_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_reset_time > reset_interval:
        pause_count = 0
        not_understood_count = 0
        not_audible_count = 0
        last_reset_time = current_time

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])

            X = np.array(row).reshape(1, -1)
            y_pred = model.predict(X)
            gesture_name = label_encoder.inverse_transform(y_pred)[0]

            if gesture_name.lower() == "pause":
                pause_count += 1
            elif gesture_name.lower() == "not_understood":
                not_understood_count += 1
            elif gesture_name.lower() == "not_audible":
                not_audible_count += 1

            # Trigger alerts
            if pause_count > threshold:
                ctypes.windll.user32.MessageBoxW(0, "Multiple students requested PAUSE", "Classroom Alert", 1)
                pause_count = 0 

            if not_understood_count > threshold:
                ctypes.windll.user32.MessageBoxW(0, "Multiple students said NOT UNDERSTOOD", "Classroom Alert", 1)
                not_understood_count = 0

            if not_audible_count > threshold:
                ctypes.windll.user32.MessageBoxW(0, "Multiple students said NOT AUDIBLE", "Classroom Alert", 1)
                not_audible_count = 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
