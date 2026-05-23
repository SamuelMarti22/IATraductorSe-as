import csv
import random
from pathlib import Path

# Proporciones de la división
TRAIN = 0.70
VAL   = 0.15
TEST  = 0.15

def split_dataset(dataset_folder, output_csv):

    dataset_path = Path(dataset_folder)
    filas = []

    # Recorrer categorías (Colors, Courtesy, Numbers)
    for categoria in sorted(dataset_path.iterdir()):
        if not categoria.is_dir():
            continue

        # Recorrer señas (hola, gracias, etc.)
        for seña in sorted(categoria.iterdir()):
            if not seña.is_dir():
                continue

            etiqueta = seña.name

            # Mezclar aleatoriamente para que el split no quede sesgado
            archivos = sorted(seña.glob("*.npy"))
            random.shuffle(archivos)

            total = len(archivos)
            n_train = int(total * TRAIN)
            n_val   = int(total * VAL)

            # Dividir en los tres grupos según las proporciones definidas arriba
            train_files = archivos[:n_train]
            val_files   = archivos[n_train:n_train + n_val]
            test_files  = archivos[n_train + n_val:]

            for archivo in train_files:
                filas.append([str(archivo), etiqueta, "train"])

            for archivo in val_files:
                filas.append([str(archivo), etiqueta, "val"])

            for archivo in test_files:
                filas.append([str(archivo), etiqueta, "test"])

    # Guardar CSV con columnas: archivo, etiqueta, split
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["archivo", "etiqueta", "split"])
        writer.writerows(filas)

    print(f"CSV guardado en: {output_csv}")
    print(f"Total de archivos: {len(filas)}")


def main():
    base = Path(__file__).parent.parent
    split_dataset(base / "dataset", base / "dataset" / "split.csv")

main()
