import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# SignDataset es la clase que lee el CSV y carga los archivos .npy uno a uno
class SignDataset(Dataset):

    def __init__(self, csv_path, split, label_to_idx):
        """
        Args:
            csv_path: ruta al split.csv
            split: "train", "val" o "test"
            label_to_idx: diccionario que mapea etiqueta -> número
        """
        self.muestras = []
        self.label_to_idx = label_to_idx

        # Leer solo las filas del split indicado
        with open(csv_path) as f:
            for fila in csv.DictReader(f):
                if fila["split"] == split:
                    self.muestras.append((fila["archivo"], fila["etiqueta"]))

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        archivo, etiqueta = self.muestras[idx]

        # Cargar secuencia de landmarks y convertir a tensor de PyTorch
        secuencia = np.load(archivo)
        secuencia = torch.tensor(secuencia, dtype=torch.float32)

        # El modelo necesita números, no texto — se convierte la etiqueta a su índice
        label = torch.tensor(self.label_to_idx[etiqueta], dtype=torch.long)

        return secuencia, label


def collate_fn(batch):
    #PADDING + MASKING
    # Los videos tienen distinto número de frames — se rellenan con ceros (padding)
    # para que todos queden del mismo largo dentro del batch
    # La máscara indica qué frames son reales (True) y cuáles son relleno (False)
    secuencias, labels = zip(*batch)

    max_len = max(s.shape[0] for s in secuencias)
    features = secuencias[0].shape[1]

    secuencias_padded = torch.zeros(len(secuencias), max_len, features)
    mascara = torch.zeros(len(secuencias), max_len, dtype=torch.bool)

    for i, secuencia in enumerate(secuencias):
        largo = secuencia.shape[0]
        secuencias_padded[i, :largo, :] = secuencia
        mascara[i, :largo] = True  # frames reales en True

    labels = torch.stack(labels)

    return secuencias_padded, mascara, labels


def get_dataloaders(csv_path, batch_size=32):
    # Construir diccionario etiqueta -> número desde el CSV
    # Ejemplo: {"hola": 0, "gracias": 1, "amarillo": 2, ...}
    etiquetas = set()
    with open(csv_path) as f:
        for fila in csv.DictReader(f):
            etiquetas.add(fila["etiqueta"])

    label_to_idx = {etiqueta: idx for idx, etiqueta in enumerate(sorted(etiquetas))}

    # Crear un dataset por cada split
    train_dataset = SignDataset(csv_path, "train", label_to_idx)
    val_dataset   = SignDataset(csv_path, "val",   label_to_idx)
    test_dataset  = SignDataset(csv_path, "test",  label_to_idx)

    # shuffle=True en train para que el modelo no aprenda el orden de los datos
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, label_to_idx
