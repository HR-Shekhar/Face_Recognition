# recognition/database.py
import os
import numpy as np
import pickle

class FaceDatabase:
    """
    Simple file-backed DB storing embeddings per person.
    For each person we store a .npy with shape (N, D) of embeddings.
    """

    def __init__(self, db_path="known_faces", threshold=0.45):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.threshold = threshold
        self.index = {}  # name -> embeddings np.array
        self._load_all()

    def _load_all(self):
        for fname in os.listdir(self.db_path):
            if not fname.endswith(".npy"):
                continue
            name = fname[:-4]
            arr = np.load(os.path.join(self.db_path, fname))
            self.index[name] = arr

    def save_person(self, name, embeddings):
        """
        embeddings: list or array of vectors (.shape = (k, D))
        """
        arr = np.array(embeddings, dtype=np.float32)
        path = os.path.join(self.db_path, f"{name}.npy")
        if os.path.exists(path):
            existing = np.load(path)
            arr = np.vstack([existing, arr])
        np.save(path, arr)
        self.index[name] = np.load(path)

    def list_people(self):
        return list(self.index.keys())

    def match(self, embedding):
        """
        embedding: 1D numpy vector (normalized)
        returns: (best_name, best_score)
        """
        if embedding is None:
            return "Unknown", 0.0
        best_name = "Unknown"
        best_score = -1.0
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        for name, arr in self.index.items():
            # arr shape: (k, D)
            # ensure normalized
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            arr_n = arr / norms
            sims = arr_n.dot(emb)
            max_sim = float(np.max(sims))
            if max_sim > best_score:
                best_score = max_sim
                best_name = name
        if best_score < self.threshold:
            return "Unknown", best_score
        return best_name, best_score
