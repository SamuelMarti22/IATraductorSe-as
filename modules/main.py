import os
import cv2
import sys
from pathlib import Path
from pipeline import HandPipeline 

def recorrer_videos(pipeline, data_folder, output_path):
    """
    Recorre todos los videos en la carpeta data, organizados por categorías.
    
    Args:
        pipeline: Instancia de HandPipeline para procesar frames
        data_folder: Ruta de la carpeta data
        output_path: Ruta opcional para guardar resultados
    
    Returns:
        Lista con información de videos procesados
    """
    
    data_path = Path(data_folder)
    videos_procesados = []
    
    # Recorrer categorías principales (Colors, Courtesy, Numbers)
    for categoria in sorted(data_path.iterdir()):
        if not categoria.is_dir() or categoria.name == '__pycache__':
            continue
            
        print(f"\n{'='*60}")
        print(f"Procesando categoría: {categoria.name}")
        print(f"{'='*60}")
        
        # Recorrer subcategorías (colores, gestos, números)
        for subcategoria in sorted(categoria.iterdir()):
            if not subcategoria.is_dir():
                continue
                
            print(f"\n  → Subcategoría: {subcategoria.name}")
            
            # Buscar archivos de video (.avi, .mp4, etc)
            video_files = list(subcategoria.glob("*.avi"))
            
            print(f"     Videos encontrados: {len(video_files)}")
 
            # Procesar cada video
            for video_file in sorted(video_files):
                sequence = pipeline.process_video(video_file)

                print(sequence.shape)

                # Construir ruta única por video: dataset/Courtesy/hola/hola_001.npy
                save_path = Path(output_path) / categoria.name / subcategoria.name / video_file.stem
                save_path.parent.mkdir(parents=True, exist_ok=True)  # crear carpetas si no existen

                pipeline.save_sequence(sequence, save_path)

                print(f"     ✓ {video_file.name}")
    
    print(f"\n{'='*60}")
    print(f"Total de videos procesados: {len(videos_procesados)}")
    print(f"{'='*60}\n")
    
    return videos_procesados

def main():
    base = Path(__file__).parent.parent  # raíz del proyecto
    pipeline = HandPipeline()
    recorrer_videos(pipeline, base / "data", base / "dataset")
    pipeline.close() 


main()

    