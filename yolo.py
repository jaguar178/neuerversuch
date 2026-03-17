import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(page_title="YOLO Objekterkennung", layout="centered")

st.title("🔍 YOLO Objekterkennung")
st.write("Lade ein Bild oder Video hoch und erkenne Objekte automatisch.")

# Modell laden (einmalig)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Datei hochladen", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    # Temporäre Datei speichern
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    file_path = tfile.name

    file_type = uploaded_file.type

    # 📷 BILD
    if file_type.startswith("image"):
        st.subheader("📷 Bild-Erkennung")

        results = model(file_path)
        annotated = results[0].plot()

        st.image(annotated, caption="Erkannte Objekte", use_column_width=True)

    # 🎥 VIDEO
    elif file_type.startswith("video"):
        st.subheader("🎥 Video-Erkennung (kann etwas dauern)")

        cap = cv2.VideoCapture(file_path)

        # Video Eigenschaften
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = file_path + "_output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            out.write(annotated)

            current_frame += 1
            progress_bar.progress(min(current_frame / frame_count, 1.0))

        cap.release()
        out.release()

        st.success("✅ Video verarbeitet!")

        # Video anzeigen
        with open(output_path, "rb") as f:
            st.video(f.read())

    else:
        st.error("Dateityp nicht unterstützt ❌")

    # Aufräumen
    try:
        os.remove(file_path)
    except:
        pass
