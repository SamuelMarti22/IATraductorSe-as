import cv2
from cv2.gapi import video

def procesarVideo(rutaVideo):
    # Abrir video
    cap = cv2.VideoCapture(rutaVideo)

    frame_number = 0

    while cap.isOpened():
    
        # Leer frame
        success, frame = cap.read()
    
        # Si no hay más frames
        if not success:
            break
        
        print(f"Frame {frame_number}")
    
        # frame es una imagen (numpy array)
        print(frame.shape)
    
        frame_number += 1
    
    # Liberar video
    cap.release()