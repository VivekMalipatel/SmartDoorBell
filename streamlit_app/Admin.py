import streamlit as st
import cv2
import numpy as np
import json
from pathlib import Path
from app.config import UNKNOWN_DIR, UNKNOWN_EMB_FILE, UNKNOWN_META_FILE, IMAGES_DIR, SIM_THRESHOLD, UNKNOWN_DUP_SIM
from app.faiss_store import FaissStore
from app.config import INDEX_FILE, IDS_FILE, PERSONS_FILE
from app.enroll import build_index

st.set_page_config(page_title='SmartDoorBell - Admin', layout='wide')

@st.cache_data
def load_unknowns():
    embs = None
    meta = []
    if Path(UNKNOWN_EMB_FILE).exists():
        embs = np.load(UNKNOWN_EMB_FILE)
    else:
        embs = np.zeros((0,512), dtype='float32')
    if Path(UNKNOWN_META_FILE).exists():
        with open(UNKNOWN_META_FILE,'r') as f:
            meta = json.load(f)
    return embs, meta

@st.cache_data
def load_store_snapshot():
    store = FaissStore(INDEX_FILE, IDS_FILE, PERSONS_FILE, 512)
    store.load()
    return store.persons

unknown_embs, unknown_meta = load_unknowns()
persons = load_store_snapshot()

st.header('Thresholds')
new_sim = st.slider('Match Threshold', 0.1, 0.9, float(SIM_THRESHOLD), 0.01)
new_dup = st.slider('Unknown Dedup Threshold', 0.5, 0.99, float(UNKNOWN_DUP_SIM), 0.01)
st.caption('Restart app to apply; for runtime dynamic adjust implement config persistence.')

st.header('Add New Person')
name = st.text_input('Person Name')
photos = st.file_uploader('Upload Photos', type=['jpg','jpeg','png'], accept_multiple_files=True)
add_btn = st.button('Add Person')
msg = st.empty()

if add_btn and name and photos:
    person_dir = IMAGES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in photos:
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        outp = person_dir / f.name
        cv2.imwrite(str(outp), img)
        saved += 1
    if saved>0:
        build_index()
        msg.success(f'Added {name} with {saved} photos')

st.divider()
st.header('Manage Persons')
if not persons:
    st.info('No persons enrolled')
else:
    for k,v in persons.items():
        cols = st.columns([3,1])
        with cols[0]:
            st.write(f"Label {k}: {v.get('person_id')} ({v.get('name')})")
        with cols[1]:
            if st.button(f"Remove {k}"):
                store = FaissStore(INDEX_FILE, IDS_FILE, PERSONS_FILE, 512)
                store.load()
                store.remove_person(v.get('person_id'))
                st.experimental_rerun()

st.divider()
st.header('Label Unknown Faces')
if len(unknown_meta)==0:
    st.info('No unknown faces collected yet')
else:
    cols = st.columns(5)
    for i,m in enumerate(unknown_meta):
        imgp = UNKNOWN_DIR / m['file']
        if not imgp.exists():
            continue
        img = cv2.imread(str(imgp))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with cols[i%5]:
            if st.button(f'Select #{i}'):
                st.session_state['selected_unknown'] = i
            st.image(img_rgb, caption=f'#{i}')
    si = st.session_state.get('selected_unknown', None)
    if si is not None and si < len(unknown_meta):
        st.subheader(f'Selected Unknown #{si}')
        new_name = st.text_input('Label as (existing or new person)')
        if st.button('Apply Label') and new_name:
            src_file = UNKNOWN_DIR / unknown_meta[si]['file']
            if src_file.exists():
                person_dir = IMAGES_DIR / new_name
                person_dir.mkdir(parents=True, exist_ok=True)
                dst = person_dir / src_file.name
                src_img = cv2.imread(str(src_file))
                if src_img is not None:
                    cv2.imwrite(str(dst), src_img)
                build_index()
                try:
                    src_file.unlink()
                except Exception:
                    pass
                del unknown_meta[si]
                with open(UNKNOWN_META_FILE,'w') as f:
                    json.dump(unknown_meta, f)
                st.session_state['selected_unknown'] = None
                st.success('Labeled, indexed and cleaned')
