import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = str(Path(__file__).parent.parent / "test_model" / "hand_landmarker.task")


class HandPipeline:

    def __init__(self,
                 max_num_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.hands = vision.HandLandmarker.create_from_options(options)

    # Procesar un solo frame y generar vector de landmarks para ambas manos
    def process_frame(self, frame):

        # Convertir a RGB (MediaPipe requiere RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hands.detect(mp_image)

        # Inicializar vectores para ambas manos en 0
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        # Si detecta alguna mano, procesar landmarks
        if results.hand_landmarks and results.handedness:

            # recorrer manos detectadas
            for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness):

                # Left / Right
                label = handedness[0].category_name

                hand_vector = []

                # recorrer landmarks
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

        #Abre el video
        cap = cv2.VideoCapture(video_path)

        sequence = []

        while cap.isOpened():

            # Tomar un frame del video
            success, frame = cap.read()

            if not success:
                break

            # Procesar el frame para obtener el vector de landmarks
            frame_vector = self.process_frame(frame)

            sequence.append(frame_vector)

        cap.release()

        return np.array(sequence, dtype=np.float32)

    # Guardar secuencia de vectores en un archivo .npy
    def save_sequence(self, sequence, output_path):

        np.save(output_path, sequence)

        print(f"Secuencia guardada en: {output_path}")

    def close(self):

        self.hands.close()
