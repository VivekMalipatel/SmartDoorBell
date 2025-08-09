import os
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
CONFIG_FILE = DATA_DIR / 'config.json'
FAISS_DIR = DATA_DIR / 'FAISS'
IMAGES_DIR = DATA_DIR / 'Images'
INDEX_FILE = FAISS_DIR / 'index.faiss'
IDS_FILE = FAISS_DIR / 'ids.npy'
PERSONS_FILE = FAISS_DIR / 'persons.json'
VECTORS_FILE = FAISS_DIR / 'vectors.npy'
UNKNOWN_DIR = IMAGES_DIR / 'unknown'
UNKNOWN_EMB_FILE = FAISS_DIR / 'unknown_embeddings.npy'
UNKNOWN_META_FILE = FAISS_DIR / 'unknown_meta.json'
DET_SIZE = (640, 640)
SIM_THRESHOLD = 0.4
TOPK = 1
UNKNOWN_DUP_SIM = 0.8
PROVIDERS = ['CPU']
LOG_LEVEL = 'INFO'
#TODO: dynamic config reload
if CONFIG_FILE.exists():
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
        SIM_THRESHOLD = float(cfg.get('SIM_THRESHOLD', SIM_THRESHOLD))
        TOPK = int(cfg.get('TOPK', TOPK))
        UNKNOWN_DUP_SIM = float(cfg.get('UNKNOWN_DUP_SIM', UNKNOWN_DUP_SIM))
        PROVIDERS = cfg.get('PROVIDERS', PROVIDERS)
        LOG_LEVEL = cfg.get('LOG_LEVEL', LOG_LEVEL)
    except Exception:
        pass
SIM_THRESHOLD = float(os.getenv('SIM_THRESHOLD', SIM_THRESHOLD))
TOPK = int(os.getenv('TOPK', TOPK))
UNKNOWN_DUP_SIM = float(os.getenv('UNKNOWN_DUP_SIM', UNKNOWN_DUP_SIM))
ENV_PROVIDERS = os.getenv('INSIGHTFACE_PROVIDERS')
if ENV_PROVIDERS:
    PROVIDERS = ENV_PROVIDERS.split(',')
LOG_LEVEL = os.getenv('LOG_LEVEL', LOG_LEVEL)
LEVELS = {'DEBUG':10,'INFO':20,'WARN':30,'ERROR':40}
CURRENT_LEVEL = LEVELS.get(LOG_LEVEL.upper(),20)

def log(level, msg):
    if LEVELS.get(level.upper(), 100) >= CURRENT_LEVEL:
        print(f"[{level.upper()}] {msg}")
