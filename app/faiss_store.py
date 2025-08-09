import os
import json
import numpy as np
import faiss
from pathlib import Path
from .config import VECTORS_FILE, CONFIG_FILE
import threading

class FaissStore:
    def __init__(self, index_path, ids_path, persons_path, dim):
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.persons_path = Path(persons_path)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids = np.empty((0,), dtype=np.int32)
        self.persons = {}
        self.vectors = np.zeros((0, dim), dtype='float32')
        self._loaded = False
        self._lock = threading.RLock()

    def log(self, msg):
        print(f"[FaissStore] {msg}")

    def load(self):
        with self._lock:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.dim = self.index.d
            if self.ids_path.exists():
                self.ids = np.load(self.ids_path)
            if self.persons_path.exists():
                with open(self.persons_path, 'r') as f:
                    self.persons = json.load(f)
            if Path(VECTORS_FILE).exists():
                self.vectors = np.load(VECTORS_FILE)
                if self.vectors.shape[0] != self.index.ntotal or self.vectors.shape[1] != self.dim:
                    self.vectors = np.zeros((self.index.ntotal, self.dim), dtype='float32')
            else:
                self.vectors = np.zeros((self.index.ntotal, self.dim), dtype='float32')
            self._loaded = True
            self._persist_dim()

    def _persist_dim(self):
        data = {}
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
            except Exception:
                data = {}
        data['EMBED_DIM'] = self.dim
        tmp = CONFIG_FILE.with_suffix('.tmp')
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data))
        os.replace(tmp, CONFIG_FILE)

    def rebuild(self, dim):
        with self._lock:
            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)
            self.ids = np.empty((0,), dtype=np.int32)
            self.vectors = np.zeros((0, dim), dtype='float32')
            self._persist_dim()

    def _atomic_write_json(self, path, data):
        tmp = path.with_suffix(path.suffix + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def _atomic_write_npy(self, path, arr):
        tmp = path.with_suffix(path.suffix + '.tmp')
        np.save(tmp, arr)
        os.replace(tmp, path)

    def save(self):
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            self._atomic_write_npy(self.ids_path, self.ids)
            self._atomic_write_json(self.persons_path, self.persons)
            self._atomic_write_npy(Path(VECTORS_FILE), self.vectors)
            self._persist_dim()

    def add(self, vectors, labels):
        if len(vectors)==0:
            return
        xb = np.asarray(vectors, dtype='float32')
        if xb.shape[1] != self.index.d:
            return
        with self._lock:
            self.index.add(xb)
            self.ids = np.concatenate([self.ids, np.asarray(labels, dtype=np.int32)])
            self.vectors = np.vstack([self.vectors, xb])

    def search(self, vectors, k=1):
        if self.index.ntotal == 0:
            return np.empty((0,k), dtype='float32'), np.empty((0,k), dtype=np.int32)
        xq = np.asarray(vectors, dtype='float32')
        sims, idxs = self.index.search(xq, k)
        labels = np.where(idxs>=0, self.ids[idxs], -1)
        return sims, labels

    def next_label(self):
        if len(self.persons)==0:
            return 0
        return max(int(x) for x in self.persons.keys())+1

    def register_person(self, label, person_id, name=None, resolve='keep'):
        with self._lock:
            for k,v in self.persons.items():
                if v.get('person_id')==person_id:
                    if resolve=='keep':
                        return int(k)
                    if resolve=='rename':
                        person_id = f"{person_id}_new"
                    if resolve=='replace':
                        self.persons[k] = {"person_id": person_id, "name": name}
                        return int(k)
            self.persons[str(label)] = {"person_id": person_id, "name": name}
            return label

    def remove_person(self, person_id):
        with self._lock:
            target_labels = [int(k) for k,v in self.persons.items() if v.get('person_id')==person_id]
            if not target_labels:
                return False
            mask = ~np.isin(self.ids, target_labels)
            self.ids = self.ids[mask]
            self.vectors = self.vectors[mask]
            keep_index = faiss.IndexFlatIP(self.dim)
            keep_index.add(self.vectors)
            self.index = keep_index
            for k in list(self.persons.keys()):
                if int(k) in target_labels:
                    del self.persons[k]
            self.save()
            return True

    def prune_orphans(self):
        with self._lock:
            valid_labels = set(int(k) for k in self.persons.keys())
            mask = np.array([lid in valid_labels for lid in self.ids])
            if mask.all():
                return 0
            self.ids = self.ids[mask]
            self.vectors = self.vectors[mask]
            new_index = faiss.IndexFlatIP(self.dim)
            new_index.add(self.vectors)
            self.index = new_index
            self.save()
            return 1
