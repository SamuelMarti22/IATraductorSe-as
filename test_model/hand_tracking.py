import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Crear detector
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# Abrir cámara
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Efecto espejo
    frame = cv2.flip(frame, 1)

    # Convertir BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir a formato MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detectar manos
    detection_result = detector.detect(mp_image)

    # Dibujar landmarks
    if detection_result.hand_landmarks:

        for hand_landmarks in detection_result.hand_landmarks:

            h, w, _ = frame.shape

            for idx, landmark in enumerate(hand_landmarks):

                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z

    print(f"Punto {idx}: X={x}, Y={y}, Z={z}")

    # Mostrar cámara
    cv2.imshow("Hand Tracking", frame)

    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()