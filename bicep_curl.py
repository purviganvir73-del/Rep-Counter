from ultralytics import YOLO
import cv2
import math
import threading
import pyttsx3
import numpy as np

model = YOLO("yolov8n-pose.pt")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_async(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    t = threading.Thread(target=run)
    t.daemon = True
    t.start()


def calculate_angle(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    BA = A - B
    BC = C - B

    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle

cap = cv2.VideoCapture(0)

reps = 0
stage = "up"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        frame = r.plot()

        kp_list = r.keypoints

        if kp_list is None:
            continue

        kp = kp_list.xy[0].numpy()

        # RIGHT ARM KEYPOINTS (more stable)
        shoulder = kp[6]
        elbow = kp[8]
        wrist = kp[10]

        angle = calculate_angle(shoulder, elbow, wrist)

        # REP LOGIC
        if angle < 65 and stage == "up":
            stage = "down"

        if angle > 155 and stage == "down":
            stage = "up"
            reps += 1
            speak_async(str(reps))

        # UI PANEL
        cv2.rectangle(frame, (0,0), (260,160), (0,0,0), -1)

        cv2.putText(frame, f"REPS: {reps}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        cv2.putText(frame, f"ANGLE: {int(angle)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Right Arm Curl Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
