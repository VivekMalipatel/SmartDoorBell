import numpy as np
from .config import PROVIDERS, DET_SIZE

class FaceEngine:
    def __init__(self, model_name='buffalo_l', det_size=DET_SIZE, providers=None):
        import insightface
        if providers is None:
            providers = PROVIDERS
        self.app = insightface.app.FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
    def detect(self, img):
        return self.app.get(img)
    def extract(self, img):
        faces = self.detect(img)
        embs = []
        for f in faces:
            v = getattr(f, 'embedding', None)
            if v is None:
                continue
            v = v.astype('float32')
            n = np.linalg.norm(v)
            if n>0:
                v /= n
            embs.append(v)
        return faces, embs
    def embedding_dim(self):
        test_img = np.zeros((10,10,3), dtype=np.uint8)
        faces, embs = self.extract(test_img)
        if embs:
            return len(embs[0])
        # fallback common dims
        return 512
