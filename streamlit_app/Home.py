import streamlit as st
import cv2
import time
from app.recognize import Recognizer

st.set_page_config(page_title='SmartDoorBell - Live', layout='wide')

rec = Recognizer()
CAM_SOURCES = [0]

@st.cache_resource
def get_caps():
    caps = []
    for src in CAM_SOURCES:
        cap = cv2.VideoCapture(src)
        caps.append(cap)
    return caps

caps = get_caps()
cols = st.columns(len(caps))
run = st.checkbox('Run', value=True)
placeholder = st.empty()

while run:
    for cap, col in zip(caps, cols):
        ok, frame = cap.read()
        if not ok:
            continue
        annotated, faces, names, embs = rec.process_frame(frame)
        col.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels='RGB', use_column_width=True)
    time.sleep(0.05)
else:
    for cap in caps:
        cap.release()
