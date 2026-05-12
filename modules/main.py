import os
import cv2
from pathlib import Path
import pipeline

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

                pipeline.save_sequence(
                    sequence,
                    output_path
                )

                pipeline.close()
                print(f"     ✓ {video_file.name}")
    
    print(f"\n{'='*60}")
    print(f"Total de videos procesados: {len(videos_procesados)}")
    print(f"{'='*60}\n")
    
    return videos_procesados

def __main__:
    pipeline = HandPipeline()
    recorrer_videos(pipeline, "../data", "../dataset")

    