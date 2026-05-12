import cv2
import mediapipe as mp
import numpy as np


class HandPipeline:

    def __init__(self,
                 max_num_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    # Procesar un solo frame y generar vector de landmarks para ambas manos
    def process_frame(self, frame):

        # Convertir a RGB (MediaPipe requiere RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        results = self.hands.process(rgb)

        # Inicializar vectores para ambas manos en 0 
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        # Si detecta alguna mano, procesar landmarks
        if results.multi_hand_landmarks and results.multi_handedness:

            # recorrer manos detectadas
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness):

                # Left / Right
                label = handedness.classification[0].label

                hand_vector = []

                # recorrer landmarks
                for landmark in hand_landmarks.landmark:

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