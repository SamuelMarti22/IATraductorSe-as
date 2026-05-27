import os
import csv
import sys
import cv2
import torch
import tempfile
import numpy as np
import mediapipe as mp
import gradio as gr
from pathlib import Path

# Variables de entorno para que MediaPipe funcione en Lightning AI sin GPU
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ""
os.environ["EGL_PLATFORM"] = "surfaceless"

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import HandPipeline
from model import SignLSTM
from dataloader import normalizar_landmarks


base = Path(__file__).parent.parent

# Conexiones entre landmarks para dibujar el esqueleto de la mano
CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def cargar_etiquetas(csv_path):
    etiquetas = set()
    with open(csv_path) as f:
        for fila in csv.DictReader(f):
            etiquetas.add(fila["etiqueta"])
    label_to_idx = {e: i for i, e in enumerate(sorted(etiquetas))}
    idx_to_label = {i: e for e, i in label_to_idx.items()}
    return idx_to_label


def dibujar_landmarks(frame, landmarks, color_puntos, color_lineas):
    h, w = frame.shape[:2]
    puntos = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Dibujar líneas del esqueleto
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], color_lineas, 2)

    # Dibujar puntos
    for punto in puntos:
        cv2.circle(frame, punto, 5, color_puntos, -1)


# Cargar modelo y pipeline una sola vez al iniciar
idx_to_label = cargar_etiquetas(base / "dataset" / "split.csv")
num_classes   = len(idx_to_label)

model = SignLSTM(input_size=126, hidden_size=256, num_layers=2,
                 num_classes=num_classes, dropout=0.0)
model.load_state_dict(torch.load(base / "modelo_entrenado.pth", map_location="cpu"))
model.eval()

pipeline = HandPipeline()


def predecir(video_path):
    if video_path is None:
        return None, "No se recibió video."

    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Archivo temporal para guardar el video anotado
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    secuencia      = []
    frames_con_mano = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detectar landmarks con MediaPipe
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = pipeline.hands.detect(mp_image)

        # Construir vector de landmarks igual que en el pipeline
        left_hand  = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        if results.hand_landmarks and results.handedness:
            frames_con_mano += 1
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                label = handedness[0].category_name
                vec   = []
                for lm in hand_landmarks:
                    vec.extend([lm.x, lm.y, lm.z])
                vec = np.array(vec, dtype=np.float32)
                if label == "Left":
                    left_hand = vec
                    dibujar_landmarks(frame, hand_landmarks, (255, 80, 80), (200, 50, 50))
                elif label == "Right":
                    right_hand = vec
                    dibujar_landmarks(frame, hand_landmarks, (80, 255, 80), (50, 200, 50))

        secuencia.append(np.concatenate([left_hand, right_hand]))
        writer.write(frame)

    cap.release()
    writer.release()

    if len(secuencia) == 0 or frames_con_mano == 0:
        return tmp.name, "No se detectaron manos en el video."

    # Predecir con el modelo
    secuencia = torch.tensor(np.array(secuencia), dtype=torch.float32)
    secuencia = normalizar_landmarks(secuencia)
    secuencia = secuencia.unsqueeze(0)
    mascara   = torch.ones(1, secuencia.shape[1], dtype=torch.bool)

    with torch.no_grad():
        outputs = model(secuencia, mascara)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred_idx = probs.max(dim=1)

    prediccion = idx_to_label[pred_idx.item()]
    confianza  = conf.item()

    top3_probs, top3_idx = probs[0].topk(3)
    resultado = f"### {prediccion} ({confianza:.0%})\n\n**Top 3:**\n"
    for prob, idx in zip(top3_probs, top3_idx):
        resultado += f"- {idx_to_label[idx.item()]}: {prob.item():.0%}\n"
    resultado += f"\n*Frames procesados: {len(secuencia[0])} | Con mano detectada: {frames_con_mano}*"

    return tmp.name, resultado


demo = gr.Interface(
    fn=predecir,
    inputs=gr.Video(label="Graba tu seña aquí"),
    outputs=[
        gr.Video(label="Video con landmarks"),
        gr.Markdown(label="Predicción"),
    ],
    title="Traductor de Señas",
    description="Graba un gesto con tu mano. Se mostrarán los puntos detectados y la predicción del modelo.",
)

demo.launch(share=True)
