import os
import sys
import csv
from pathlib import Path

import cv2
import torch
import numpy as np
import mediapipe as mp
import gradio as gr

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
# Base path
# =========================================================

base = Path(__file__).parent.parent

# =========================================================
# Pipeline MediaPipe
# =========================================================

pipeline = HandPipeline()

# =========================================================
# Cargar etiquetas
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
    hidden_size=256,
    num_layers=2,
    num_classes=num_classes,
    dropout=0.0
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
# Procesar video completo
# =========================================================

def procesar_video(video_path):

    if video_path is None:
        return "No se recibió video."

    cap = cv2.VideoCapture(video_path)

    secuencia = []

    total_frames = 0
    frames_con_mano = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        # ================================================
        # Resize para acelerar procesamiento
        # ================================================

        frame = cv2.resize(
            frame,
            (640, 480)
        )

        # ================================================
        # MediaPipe
        # ================================================

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

        # ================================================
        # Contar frames con mano
        # ================================================

        if results.hand_landmarks:
            frames_con_mano += 1

        # ================================================
        # Extraer landmarks
        # ================================================

        landmarks_vector = extraer_landmarks(
            results
        )

        secuencia.append(
            landmarks_vector
        )

    cap.release()

    # =====================================================
    # Validación
    # =====================================================

    if len(secuencia) == 0:

        return "No se encontraron frames."

    if frames_con_mano == 0:

        return "No se detectaron manos."

    # =====================================================
    # Inferencia
    # =====================================================

    prediccion, confianza = inferir_secuencia(
        secuencia
    )

    # =====================================================
    # Resultado
    # =====================================================

    resultado = f"""
# Predicción

## {prediccion}

Confianza: {confianza:.0%}

Frames totales: {total_frames}

Frames con mano detectada: {frames_con_mano}
"""

    return resultado

# =========================================================
# UI
# =========================================================

with gr.Blocks() as demo:

    gr.Markdown(
        """
# Traductor de Señas

Graba una seña corta usando tu webcam y luego presiona "Traducir".
"""
    )

    video_input = gr.Video(
        sources=["webcam"],
        label="Grabar seña"
    )

    boton = gr.Button(
        "Traducir"
    )

    salida = gr.Markdown()

    boton.click(
        fn=procesar_video,
        inputs=video_input,
        outputs=salida
    )

# =========================================================
# Launch
# =========================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.getenv(
                "PORT",
                os.getenv(
                    "GRADIO_SERVER_PORT",
                    8080
                )
            )
        ),
        share=False
    )