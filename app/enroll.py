import cv2
from .config import IMAGES_DIR, FAISS_DIR, INDEX_FILE, IDS_FILE, PERSONS_FILE, DET_SIZE
from .embedder import Embedder
from .faiss_store import FaissStore


def build_index():
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    emb = Embedder(det_size=DET_SIZE)
    store = FaissStore(INDEX_FILE, IDS_FILE, PERSONS_FILE, 512)
    store.load()
    new_vectors = []
    new_labels = []
    for pid_dir in sorted(IMAGES_DIR.glob('*')):
        if not pid_dir.is_dir() or pid_dir.name == 'unknown':
            continue
        pid = pid_dir.name
        existing_label = None
        for k,v in store.persons.items():
            if v.get('person_id')==pid:
                existing_label = int(k)
                break
        label = existing_label if existing_label is not None else store.next_label()
        for imgp in pid_dir.glob('*'):
            if imgp.suffix.lower() not in ('.jpg','.jpeg','.png'):
                continue
            img = cv2.imread(str(imgp))
            if img is None:
                continue
            _, embs = emb.embed_faces(img)
            for v in embs:
                if emb.dim and emb.dim != store.dim and store.index.ntotal==0:
                    store.rebuild(emb.dim)
                if v.shape[0] != store.dim:
                    continue
                new_vectors.append(v)
                new_labels.append(label)
        if existing_label is None:
            store.register_person(label, pid, name=pid)
    store.add(new_vectors, new_labels)
    store.save()
    return True
