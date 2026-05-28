import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import gradio as gr
import numpy as np

# =========================================================
# Configuración para MediaPipe en servidores Linux/headless
# =========================================================

os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ""
os.environ["EGL_PLATFORM"] = "surfaceless"

# =========================================================
# Imports del proyecto
# =========================================================

sys.path.insert(0, str(Path(__file__).parent))

from preprocess.pipeline import HandPipeline

# =========================================================
# Conexiones entre landmarks
# =========================================================

CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# =========================================================
# Inicializar pipeline UNA sola vez
# =========================================================

pipeline = HandPipeline()

# =========================================================
# Variables de estado para buffer de landmarks y última predicción
# =========================================================

BUFFER_FRAMES = 15

buffer_landmarks = []

ultima_prediccion = "Esperando seña..."

# =========================================================
# Dibujar landmarks
# =========================================================

def dibujar_landmarks(frame, landmarks, color_puntos, color_lineas):

    h, w = frame.shape[:2]

    puntos = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ]

    # Dibujar líneas
    for a, b in CONEXIONES:
        cv2.line(
            frame,
            puntos[a],
            puntos[b],
            color_lineas,
            2
        )

    # Dibujar puntos
    for punto in puntos:
        cv2.circle(
            frame,
            punto,
            5,
            color_puntos,
            -1
        )

# =========================================================
# Procesamiento frame por frame
# =========================================================

def process_frame(video_frame):

    if video_frame is None:
        return None

    global buffer_landmarks
    global ultima_prediccion

    # Gradio entrega RGB
    # OpenCV trabaja normalmente en BGR
    frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))

    # MediaPipe necesita RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detectar manos
    results = pipeline.hands.detect(mp_image)

    landmarks_vector = extraer_landmarks(results)

    buffer_landmarks.append(landmarks_vector)

    # Dibujar landmarks si hay manos
    if results.hand_landmarks and results.handedness:

        for hand_landmarks, handedness in zip(
            results.hand_landmarks,
            results.handedness
        ):

            label = handedness[0].category_name

            # Mano izquierda
            if label == "Left":

                dibujar_landmarks(
                    frame,
                    hand_landmarks,
                    (255, 80, 80),
                    (200, 50, 50)
                )

            # Mano derecha
            elif label == "Right":

                dibujar_landmarks(
                    frame,
                    hand_landmarks,
                    (80, 255, 80),
                    (50, 200, 50)
                )

    progress = len(buffer_landmarks) / BUFFER_FRAMES


    # =====================================================
    # DIBUJAR BARRA DE PROGRESO
    # =====================================================

    bar_width = int(frame.shape[1] * progress)

    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 30),
        (bar_width, frame.shape[0]),
        (0, 255, 0),
        -1
    )

    # =====================================================
    # TEXTO EN PANTALLA
    # =====================================================

    cv2.putText(
        frame,
        ultima_prediccion,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # =====================================================
    # CUANDO EL BUFFER SE LLENA
    # =====================================================

    if len(buffer_landmarks) >= BUFFER_FRAMES:

        ultima_prediccion = "Procesando..."

        print("Aquí irá la inferencia")

        buffer_landmarks = []

    # Regresar RGB para Gradio
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# =========================================================
# Extraer landmarks en un vector de 126 floats (63 por mano)
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

    return np.concatenate([left_hand, right_hand])

# =========================================================
# UI
# =========================================================

def build_ui():

    with gr.Blocks() as demo:

        gr.Markdown("# Traductor de señas — Realtime landmarks")

        gr.Markdown(
            """
            Activa la webcam y mueve tu mano frente a la cámara.
            Los landmarks de MediaPipe se dibujarán en tiempo real.
            """
        )

        with gr.Row():

            cam = gr.Image(
                height=360,
                label="Webcam",
                sources=["webcam"],
                streaming=True
            )

            out = gr.Image(
                height=360,
                label="Landmarks detectados"
            )

        # Stream en tiempo real
        cam.stream(
            fn=process_frame,
            inputs=[cam],
            outputs=out
        )

    return demo

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.getenv(
                "PORT",
                os.getenv("GRADIO_SERVER_PORT", 8081)
            )
        )
    )

    args = parser.parse_args()

    ui = build_ui()

    ui.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False
    )