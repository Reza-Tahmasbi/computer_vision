import math
import numpy as np
import cv2
import mediapipe as mp
import sys
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

video_path = "../../assets/vids/athlete.mp4"
vid = cv2.VideoCapture(video_path)
vid.set(3, 200)
vid.set(4, 200)

with mp_pose.Pose(
    static_image_mode=False,  # Change to False for video
    min_detection_confidence=0.5,
    model_complexity=0 # 0: Lite, 1: Full, 2: Heavy
) as pose:

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        # frame = detect_ball_by_contour(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

vid.release()
cv2.destroyAllWindows()
