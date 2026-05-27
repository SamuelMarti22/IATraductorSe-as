import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
# Se importa la API nueva de MediaPipe (Tasks API) ya que muestra un mejor funcionamiento en este tipo de servidores
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ruta al modelo de detección de manos que usa MediaPipe Tasks API
# El modelo viene en un archivo separado (.task) 
# Path(__file__) obtiene la ruta de este archivo sin importar desde dónde se corra
MODEL_PATH = str(Path(__file__).parent.parent.parent / "test_model" / "hand_landmarker.task")


class HandPipeline:

    def __init__(self,
                 max_num_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):

        # Configurar las opciones del detector con la API (Tasks API)
        # Se carga el modelo desde el archivo .task
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        # Crear el detector de manos con las opciones configuradas
        self.hands = vision.HandLandmarker.create_from_options(options)

    def normalizar_landmarks(self, secuencia):
        # 1. Resta la muñeca (posición) → gesto centrado sin importar dónde esté la mano
        # 2. Divide por tamaño de mano (escala) → gesto invariante a distancia de la cámara
        T = secuencia.shape[0]

        for inicio in [0, 63]:
            hand = secuencia[:, inicio:inicio + 63].reshape(T, 21, 3)
            wrist = hand[:, 0:1, :]
            detectada = (np.abs(wrist).sum(axis=2, keepdims=True) > 0)  # (T, 1, 1)

            # Centrar en la muñeca
            hand = hand - wrist * detectada

            # Normalizar escala: distancia muñeca → base del dedo medio (landmark 9)
            escala = np.linalg.norm(hand[:, 9:10, :], axis=2, keepdims=True)
            escala = np.clip(escala, 1e-6, None)
            hand = np.where(detectada, hand / escala, hand)

            secuencia[:, inicio:inicio + 63] = hand.reshape(T, 63)

        return secuencia


    # Procesar un solo frame y generar vector de landmarks para ambas manos
    def process_frame(self, frame):

        # Convertir a RGB (MediaPipe requiere RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertir la imagen al formato que espera la API nueva de MediaPipe
        # Se envuelve en un objeto mp.Image antes de detectar
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hands.detect(mp_image)

        # Inicializar vectores para ambas manos en 0
        # Si una mano no se detecta en el frame, su vector queda en ceros
        # Cada mano tiene 21 landmarks × 3 coordenadas (x, y, z) = 63 valores
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        # Si detecta alguna mano, procesar landmarks
        if results.hand_landmarks and results.handedness:

            # recorrer manos detectadas
            for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness):

                # Identificar si es mano izquierda o derecha
                # Se lee:  handedness[0].category_name
                label = handedness[0].category_name

                hand_vector = []

                # recorrer landmarks (21 puntos de la mano)
                # Cada landmark tiene coordenadas x, y, z
                for landmark in hand_landmarks:

                    hand_vector.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                hand_vector = np.array(hand_vector, dtype=np.float32)

                # guardar según mano
                if label == "Left":
                    left_hand = hand_vector

                elif label == "Right":
                    right_hand = hand_vector

        # Concatenar vector para ambas manos (63 para cada mano, total 126)
        frame_vector = np.concatenate([
            left_hand,
            right_hand
        ])

        return frame_vector

    # Procesar video completo y generar secuencia de vectores
    def process_video(self, video_path):

        # Abre el video con OpenCV — cv2.VideoCapture permite leer frame por frame
        cap = cv2.VideoCapture(str(video_path))

        sequence = []

        while cap.isOpened():

            # Tomar un frame del video
            # cap.read() avanza un frame y devuelve la imagen de ese momento
            success, frame = cap.read()

            if not success:
                break

            # Procesar el frame para obtener el vector de landmarks
            frame_vector = self.process_frame(frame)

            sequence.append(frame_vector)

        cap.release()

        # Retorna array de forma (num_frames, 126)
        # Cada fila es un frame, cada columna es un valor de landmark
        sequence = np.array(sequence, dtype=np.float32)
        sequence = self.normalizar_landmarks(sequence)
        return sequence

    # Guardar secuencia de vectores en un archivo .npy
    # Se usa .npy en lugar de .txt porque es más rápido de leer y ocupa menos espacio
    def save_sequence(self, sequence, output_path):

        np.save(output_path, sequence)

        print(f"Secuencia guardada en: {output_path}")

    def close(self):

        self.hands.close()
