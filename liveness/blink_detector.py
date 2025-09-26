# liveness/blink_detector.py
import cv2
import numpy as np
import mediapipe as mp
import time

class BlinkDetector:
    def __init__(self, blink_threshold=0.25, consecutive_frames=3, liveness_timeout=3):
        self.blink_threshold = blink_threshold
        self.consecutive_frames = consecutive_frames
        self.counter = 0
        self.last_blink_time = 0
        self.liveness_timeout = liveness_timeout

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    def _eye_aspect_ratio(self, landmarks, eye_indices, frame_w, frame_h):
        coords = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
        A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        return (A + B) / (2.0 * C)

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        is_real = False

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE, w, h)
            right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < self.blink_threshold:
                self.counter += 1
            else:
                if self.counter >= self.consecutive_frames:
                    self.last_blink_time = time.time()  # update last blink time
                self.counter = 0

        # Check if last blink was recent
        if time.time() - self.last_blink_time < self.liveness_timeout:
            is_real = True

        return is_real
