from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import cv2
from ultralytics import YOLO

app = FastAPI()

# YOLO Modell laden (wird automatisch heruntergeladen)
model = YOLO("yolov8n.pt")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "YOLO API läuft 🚀"}


# 📷 Bild-Erkennung
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    output_path = os.path.join(RESULT_DIR, f"result_{file.filename}")

    # Datei speichern
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # YOLO ausführen
    results = model(input_path)

    # Ergebnisbild speichern
    annotated_frame = results[0].plot()
    cv2.imwrite(output_path, annotated_frame)

    return FileResponse(output_path)


# 🎥 Video-Erkennung
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    output_path = os.path.join(RESULT_DIR, f"result_{file.filename}")

    # Datei speichern
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(input_path)

    # Video Eigenschaften
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return FileResponse(output_path)
