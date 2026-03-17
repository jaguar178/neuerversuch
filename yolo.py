import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

model = YOLO("yolov8n.pt")

st.title("YOLO Objekterkennung")

uploaded_file = st.file_uploader("Bild oder Video hochladen")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if uploaded_file.type.startswith("image"):
        results = model(tfile.name)
        annotated = results[0].plot()
        st.image(annotated, caption="Erkanntes Bild")

    elif uploaded_file.type.startswith("video"):
        st.video(tfile.name)
        st.write("Video-Erkennung ist rechenintensiv ⚠️")
