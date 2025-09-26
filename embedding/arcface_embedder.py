# embedding/arcface_embedder.py
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

class ArcFaceEmbedder:
    """
    If you want to compute embedding from a cropped/aligned face image separately.
    Uses insightface's model (buffalo_l includes embedder).
    """

    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640,640)):
        self.ctx_id = ctx_id
        self.app = FaceAnalysis(name=model_name, providers=None)
        self.app.prepare(ctx_id=ctx_id if ctx_id is not None else -1, det_size=det_size)

    def get_embedding_from_face(self, face_bgr):
        """
        face_bgr: cropped face in BGR (aligned)
        returns: normalized numpy vector
        """
        # insightface FaceAnalysis.get() expects full images, but we can call get on the small image
        faces = self.app.get(face_bgr)
        if len(faces) == 0:
            return None
        f = faces[0]
        emb = None
        if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
            emb = np.array(f.normed_embedding, dtype=np.float32)
        elif hasattr(f, "embedding") and f.embedding is not None:
            emb = np.array(f.embedding, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
        return emb
