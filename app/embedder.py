import numpy as np
from .face_engine import FaceEngine

class Embedder:
    def __init__(self, det_size=(640,640)):
        self.engine = FaceEngine(det_size=det_size)
        self.dim = None

    def detect(self, img):
        return self.engine.detect(img)

    def embed_faces(self, img):
        faces, embs = self.engine.extract(img)
        if self.dim is None and embs:
            self.dim = len(embs[0])
        return faces, embs
