# utils/register_person.py
import cv2
import numpy as np
from detection.scrfd_detector import SCRFDDetector
from recognition.database import FaceDatabase
import time

def capture_and_register(name, num_samples=5, device_id=0, ctx_id=0):
    """
    Open webcam, capture num_samples face crops and embeddings, save to DB.
    """
    detector = SCRFDDetector(ctx_id=ctx_id)
    db = FaceDatabase()

    cap = cv2.VideoCapture(device_id)
    collected = []
    print(f"[INFO] Please position {name} in front of the camera. Capturing {num_samples} good frames...")
    t0 = time.time()
    while len(collected) < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector.detect(frame)
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)
            cv2.imshow("Register", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        # pick largest face
        faces = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
        f = faces[0]
        x1,y1,x2,y2 = f['bbox']
        emb = f['embedding']
        if emb is not None:
            collected.append(emb)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"Captured {len(collected)}/{num_samples}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(collected) == 0:
        print("[WARN] No embeddings captured.")
        return
    db.save_person(name, collected)
    print(f"[OK] Registered {name} with {len(collected)} samples in DB. Took {time.time() - t0:.1f}s")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--samples", type=int, default=5)
    args = p.parse_args()
    capture_and_register(args.name, num_samples=args.samples)
