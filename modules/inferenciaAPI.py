import os
import sys
import csv
import tempfile

from pathlib import Path

import cv2
import torch
import numpy as np
import mediapipe as mp

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# Configuración MediaPipe
# =========================================================

os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ""
os.environ["EGL_PLATFORM"] = "surfaceless"

# =========================================================
# Imports del proyecto
# =========================================================

sys.path.insert(0, str(Path(__file__).parent))

from preprocess.pipeline import HandPipeline
from model import SignLSTM

# =========================================================
# FastAPI
# =========================================================

app = FastAPI()

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Base path
# =========================================================

base = Path(__file__).parent.parent

HIDDEN_SIZE = 384
NUM_LAYERS  = 2
DROPOUT     = 0.3

# =========================================================
# Pipeline
# =========================================================

pipeline = HandPipeline()

# =========================================================
# Etiquetas
# =========================================================

def cargar_etiquetas(csv_path):

    etiquetas = set()

    with open(csv_path) as f:

        for fila in csv.DictReader(f):
            etiquetas.add(fila["etiqueta"])

    label_to_idx = {
        e: i
        for i, e in enumerate(sorted(etiquetas))
    }

    idx_to_label = {
        i: e
        for e, i in label_to_idx.items()
    }

    return idx_to_label

# =========================================================
# Modelo
# =========================================================

idx_to_label = cargar_etiquetas(
    base / "dataset" / "split.csv"
)

num_classes = len(idx_to_label)

model = SignLSTM(
    input_size=126,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
)

model.load_state_dict(
    torch.load(
        base / "modelo_entrenado.pth",
        map_location="cpu"
    )
)

model.eval()

# =========================================================
# Extraer landmarks
# =========================================================

def extraer_landmarks(results):

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.hand_landmarks and results.handedness:

        for hand_landmarks, handedness in zip(
            results.hand_landmarks,
            results.handedness
        ):

            label = handedness[0].category_name

            vec = []

            for lm in hand_landmarks:
                vec.extend([lm.x, lm.y, lm.z])

            vec = np.array(vec, dtype=np.float32)

            if label == "Left":
                left_hand = vec

            elif label == "Right":
                right_hand = vec

    return np.concatenate([
        left_hand,
        right_hand
    ])

# =========================================================
# Inferencia
# =========================================================

def inferir_secuencia(secuencia):

    secuencia = np.array(
        secuencia,
        dtype=np.float32
    )

    secuencia = pipeline.normalizar_landmarks(
        secuencia
    )

    secuencia = torch.tensor(
        secuencia,
        dtype=torch.float32
    )

    secuencia = secuencia.unsqueeze(0)

    mascara = torch.ones(
        1,
        secuencia.shape[1],
        dtype=torch.bool
    )

    with torch.no_grad():

        outputs = model(
            secuencia,
            mascara
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        conf, pred_idx = probs.max(dim=1)

    prediccion = idx_to_label[
        int(pred_idx.item())
    ]

    confianza = conf.item()

    return prediccion, confianza

# =========================================================
# Procesar video
# =========================================================

def procesar_video(video_path):

    cap = cv2.VideoCapture(video_path)

    secuencia = []

    total_frames = 0
    frames_con_mano = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        frame = cv2.resize(
            frame,
            (320, 240)
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = pipeline.hands.detect(
            mp_image
        )

        if results.hand_landmarks:
            frames_con_mano += 1

        landmarks_vector = extraer_landmarks(
            results
        )

        secuencia.append(
            landmarks_vector
        )

    cap.release()

    if len(secuencia) == 0:

        return {
            "prediccion": "Sin frames",
            "confianza": 0
        }

    if frames_con_mano == 0:

        return {
            "prediccion": "No se detectaron manos",
            "confianza": 0
        }

    prediccion, confianza = inferir_secuencia(
        secuencia
    )

    return {
        "prediccion": prediccion,
        "confianza": confianza,
        "frames_totales": total_frames,
        "frames_con_mano": frames_con_mano
    }

# =========================================================
# Endpoint
# =========================================================

@app.post("/predict")
async def predict(
    video: UploadFile = File(...)
):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".webm"
    ) as temp_video:

        content = await video.read()

        temp_video.write(content)

        temp_path = temp_video.name

    resultado = procesar_video(
        temp_path
    )

    os.remove(temp_path)

    return resultado