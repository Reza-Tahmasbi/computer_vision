import cv2
import os
import sys
import numpy as np
from openpose import pyopenpose as op

# Set OpenPose paths
dir_path = os.path.dirname(os.path.realpath(__file__))

params = {
    'model_folder': '/path/to/openpose/models/',  # <<< set your correct path
    'model_pose': 'BODY_25',
    'net_resolution': '-1x368'
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Video input
cap = cv2.VideoCapture("../../../assets/vids/athlete.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- OpenPose ---
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    output_frame = datum.cvOutputData

    # --- Ball Detection (Color-Based) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Example: Detecting an orange ball (tune these!)
    lower_color = np.array([5, 150, 150])
    upper_color = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # adjust this threshold to avoid noise
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output_frame, center, radius, (0, 255, 0), 2)
            cv2.putText(output_frame, "Ball", (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show results
    cv2.imshow("OpenPose + Ball Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
