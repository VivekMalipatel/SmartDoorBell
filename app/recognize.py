import cv2
import numpy as np
from .config import INDEX_FILE, IDS_FILE, PERSONS_FILE, SIM_THRESHOLD, TOPK, UNKNOWN_DIR, DET_SIZE, UNKNOWN_DUP_SIM, UNKNOWN_EMB_FILE, UNKNOWN_META_FILE
from .embedder import Embedder
from .faiss_store import FaissStore
import json
from pathlib import Path


class Recognizer:
    def __init__(self):
        self.emb = Embedder(det_size=DET_SIZE)
        self.store = FaissStore(INDEX_FILE, IDS_FILE, PERSONS_FILE, 512)
        self.store.load()
        self.unknown_embs = None
        self.unknown_meta = []
        self._load_unknown()

    def _ensure_unknown_dim(self, dim):
        if self.unknown_embs is None:
            self.unknown_embs = np.zeros((0,dim), dtype='float32')
        elif self.unknown_embs.shape[1] != dim:
            self.unknown_embs = np.zeros((0,dim), dtype='float32')

    def _load_unknown(self):
        if Path(UNKNOWN_EMB_FILE).exists():
            self.unknown_embs = np.load(UNKNOWN_EMB_FILE)
        else:
            self.unknown_embs = np.zeros((0,512), dtype='float32')
        if Path(UNKNOWN_META_FILE).exists():
            with open(UNKNOWN_META_FILE,'r') as f:
                self.unknown_meta = json.load(f)
        else:
            self.unknown_meta = []

    def _save_unknown(self):
        Path(UNKNOWN_EMB_FILE).parent.mkdir(parents=True, exist_ok=True)
        np.save(UNKNOWN_EMB_FILE, self.unknown_embs)
        with open(UNKNOWN_META_FILE,'w') as f:
            json.dump(self.unknown_meta, f)

    def _maybe_add_unknown(self, frame, face, emb):
        self._ensure_unknown_dim(emb.shape[0])
        if self.unknown_embs.shape[0]>0:
            sims = emb @ self.unknown_embs.T
            if sims.max() >= UNKNOWN_DUP_SIM:
                return
        box = face.bbox.astype(int)
        crop = frame[box[1]:box[3], box[0]:box[2]].copy()
        UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
        fn = UNKNOWN_DIR / f"u_{len(self.unknown_meta)}.jpg"
        cv2.imwrite(str(fn), crop)
        self.unknown_embs = np.vstack([self.unknown_embs, emb.reshape(1,-1)])
        self.unknown_meta.append({'file': fn.name})
        self._save_unknown()

    def annotate(self, frame, faces, names):
        dimg = frame.copy()
        for f, name in zip(faces, names):
            box = f.bbox.astype(int)
            color = (0,255,0) if name!='Unknown' else (0,0,255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(dimg, name, (box[0], box[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return dimg

    def process_frame(self, frame):
        faces, embs = self.emb.embed_faces(frame)
        if self.emb.dim and self.emb.dim != self.store.dim and self.store.index.ntotal==0:
            self.store.rebuild(self.emb.dim)
        names = []
        if len(embs)>0 and self.store.index.ntotal>0:
            sims, labels = self.store.search(embs, k=TOPK)
            for i, emb in enumerate(embs):
                sim = sims[i,0] if sims.shape[1]>0 else -1
                lbl = labels[i,0] if labels.shape[1]>0 else -1
                if sim >= SIM_THRESHOLD and lbl>=0:
                    meta = self.store.persons.get(str(lbl), {})
                    name = meta.get('name') or meta.get('person_id') or 'Known'
                else:
                    name = 'Unknown'
                    self._maybe_add_unknown(frame, faces[i], emb)
                names.append(name)
        else:
            names = ['Unknown']*len(faces)
        annotated = self.annotate(frame, faces, names)
        return annotated, faces, names, embs
