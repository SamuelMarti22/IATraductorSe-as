import torch
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import get_dataloaders
from model import SignLSTM


base       = Path(__file__).parent.parent
csv_path   = base / "dataset" / "split.csv"
model_path = base / "modelo_entrenado.pth"

train_loader, val_loader, test_loader, label_to_idx = get_dataloaders(csv_path, batch_size=32)
idx_to_label = {i: e for e, i in label_to_idx.items()}
num_classes  = len(label_to_idx)

model = SignLSTM(input_size=126, hidden_size=256, num_layers=2,
                 num_classes=num_classes, dropout=0.0)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Contar aciertos y totales por gesto
aciertos = defaultdict(int)
totales  = defaultdict(int)

with torch.no_grad():
    for secuencias, mascara, labels in test_loader:
        outputs = model(secuencias, mascara)
        preds   = outputs.argmax(dim=1)

        for pred, real in zip(preds, labels):
            gesto = idx_to_label[real.item()]
            totales[gesto]  += 1
            if pred.item() == real.item():
                aciertos[gesto] += 1

# Mostrar resultados ordenados de mayor a menor accuracy
print(f"\n{'Gesto':<20} {'Aciertos':>10} {'Total':>8} {'Accuracy':>10}")
print("-" * 52)

resultados = []
for gesto in sorted(totales):
    acc = aciertos[gesto] / totales[gesto] if totales[gesto] > 0 else 0
    resultados.append((acc, gesto, aciertos[gesto], totales[gesto]))

for acc, gesto, ac, tot in sorted(resultados, reverse=True):
    estado = "✓" if acc >= 0.6 else "✗"
    print(f"{gesto:<20} {ac:>10} {tot:>8} {acc:>9.0%}  {estado}")

acc_total = sum(aciertos.values()) / sum(totales.values())
print(f"\nAccuracy total en test: {acc_total:.2%}")
