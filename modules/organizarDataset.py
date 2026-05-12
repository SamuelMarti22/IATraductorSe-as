from pathlib import Path
import shutil

# Carpeta original
SOURCE = Path("../raw_data")

# Carpeta destino
DEST = Path("../data")

# Contadores por categoria + label
counter = {}

# Recorre TODOS los .avi recursivamente
for video_path in SOURCE.rglob("*.avi"):

    # Ejemplo:
    # data/Colors/1/Colores/amarillo/1.avi

    parts = video_path.parts

    # Buscar la categoría principal
    # (Colors, Courtesy, Numbers)
    category = None

    for folder in parts:
        if folder.lower() in ["colors", "courtesy", "numbers"]:
            category = folder
            break

    # La etiqueta es la carpeta padre del video
    label = video_path.parent.name.lower()

    # Saltar si no encontró categoría
    if category is None:
        continue

    # Crear estructura destino:
    # dataset/Colors/amarillo/
    target_folder = DEST / category / label
    target_folder.mkdir(parents=True, exist_ok=True)

    # Contador único por categoría + label
    key = (category, label)

    counter.setdefault(key, 0)
    counter[key] += 1

    # Nombre nuevo
    new_name = f"{label}_{counter[key]:03}.avi"

    # Ruta final
    destination = target_folder / new_name

    # Copiar archivo
    shutil.copy2(video_path, destination)

    print(f"✔ {video_path} -> {destination}")

print("\nDataset organizado 🚀")