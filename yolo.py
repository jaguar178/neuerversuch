import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(page_title="YOLO Objekterkennung")

st.title("🔍 YOLO Objekterkennung")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Bild oder Video hochladen",
    type=["jpg", "jpeg", "png", "mp4", "mov"]
)

if uploaded_file is not None:

    # 👉 WICHTIG: Datei korrekt speichern mit Endung
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.write(f"Datei gespeichert unter: {file_path}")

    # EXTRA CHECK (verhindert deinen Fehler)
    if not os.path.exists(file_path):
        st.error("Datei wurde nicht korrekt gespeichert ❌")
    else:

        # 📷 BILD
        if uploaded_file.type.startswith("image"):
            st.subheader("📷 Bild-Erkennung")

            results = model(file_path)
            annotated = results[0].plot()

            st.image(annotated, caption="Erkannt", use_column_width=True)

        # 🎥 VIDEO
        elif uploaded_file.type.startswith("video"):
            st.subheader("🎥 Video-Erkennung")

            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                st.error("Video konnte nicht geöffnet werden ❌")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                output_path = file_path + "_out.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                progress = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    annotated = results[0].plot()
                    out.write(annotated)

                    current += 1
                    if total_frames > 0:
                        progress.progress(min(current / total_frames, 1.0))

                cap.release()
                out.release()

                st.success("✅ Video fertig!")

                with open(output_path, "rb") as f:
                    st.video(f.read())
