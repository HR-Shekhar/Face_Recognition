# detection/scrfd_detector.py
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

class SCRFDDetector:
    """
    Wraps insightface.FaceAnalysis with SCRFD detector + arcface embedder (buffalo_l pack).
    """

    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640,640)):
        """
        model_name: insightface model bundle (buffalo_l includes scrfd+arcface)
        ctx_id: GPU device id (0) or -1 for CPU
        det_size: (w,h) used for detector
        """
        self.ctx_id = ctx_id
        self.app = FaceAnalysis(name=model_name, providers=None)
        # prepare: ctx_id = -1 -> CPU, >=0 -> GPU id
        self.app.prepare(ctx_id=ctx_id if ctx_id is not None else -1, det_size=det_size)

    def detect(self, frame_bgr):
        """
        frame_bgr: numpy array in BGR (OpenCV)
        returns: list of face dicts with keys: bbox (x1,y1,x2,y2), kps (5 points), det_score, embedding (numpy array)
        """
        # insightface expects BGR or RGB? it handles numpy BGR images properly
        faces = self.app.get(frame_bgr)  # returns list of Face objects
        results = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            kps = f.kps.tolist() if f.kps is not None else None
            det_score = float(f.det_score) if hasattr(f, "det_score") else None
            # some packs produce normalized embedding, stored as 'normed_embedding'
            emb = None
            if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                emb = np.array(f.normed_embedding, dtype=np.float32)
            elif hasattr(f, "embedding") and f.embedding is not None:
                # optionally normalize
                emb = np.array(f.embedding, dtype=np.float32)
                emb = emb / np.linalg.norm(emb)
            results.append({
                "bbox": bbox,
                "kps": kps,
                "det_score": det_score,
                "embedding": emb
            })
        return results
