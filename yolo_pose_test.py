from ultralytics import YOLO
import cv2

# load YOLO pose model
model = YOLO("yolov8n-pose.pt")  # auto-downloads the model first time

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run YOLO pose detection
    results = model(frame, stream=True)

    # draw skeleton/keypoints
    for r in results:
        frame = r.plot()  # yolo handles drawing everything

    cv2.imshow("Purvi YOLO Pose Tracking", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
