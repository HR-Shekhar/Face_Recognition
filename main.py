# main.py
import cv2
import numpy as np
from detection.scrfd_detector import SCRFDDetector
from recognition.database import FaceDatabase
from liveness.blink_detector import BlinkDetector

def run_live(device_id=0, ctx_id=0, det_size=(640, 640)):
    detector = SCRFDDetector(ctx_id=ctx_id, det_size=det_size)
    db = FaceDatabase()
    blink_detector = BlinkDetector(liveness_timeout=3)  # re-check every 3 sec

    print("[INFO] Known people:", db.list_people())

    cap = cv2.VideoCapture(device_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        # Run blink-based liveness
        is_real = blink_detector.detect(frame)
        liveness_status = "Real" if is_real else "Spoof?"

        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            emb = f["embedding"]
            name, sim = db.match(emb)

            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label with recognition + liveness
            label = f"{name} {sim:.2f} [{liveness_status}]"
            cv2.putText(
                frame,
                label,
                (x1, max(10, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("FaceRec - SCRFD + ArcFace + Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # set ctx_id=0 to use GPU 0; set ctx_id=-1 for CPU
    run_live(device_id=0, ctx_id=0)
