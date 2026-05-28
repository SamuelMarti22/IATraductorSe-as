import torch
import torch.nn as nn
from pathlib import Path
from dataloader import get_dataloaders
from model import SignLSTM


BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for secuencias, mascara, labels in loader:
        secuencias = secuencias.to(device)
        mascara    = mascara.to(device)
        labels     = labels.to(device)

        optimizer.zero_grad()
        outputs = model(secuencias, mascara)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for secuencias, mascara, labels in loader:
            secuencias = secuencias.to(device)
            mascara    = mascara.to(device)
            labels     = labels.to(device)

            outputs = model(secuencias, mascara)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    base       = Path(__file__).parent.parent
    csv_path   = base / "dataset" / "split.csv"
    model_path = base / "modelo_entrenado.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")

    train_loader, val_loader, test_loader, label_to_idx = get_dataloaders(csv_path, BATCH_SIZE)
    num_classes = len(label_to_idx)
    print(f"Clases: {num_classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model     = SignLSTM(input_size=126, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                         num_classes=num_classes, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Reduce el learning rate a la mitad si val_loss no mejora en 5 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mejor_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader,   criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.2%} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.2%}")

        # Guardar el modelo solo cuando mejora en validación
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  → Modelo guardado (val acc: {val_acc:.2%})")

    print(f"\nEntrenamiento terminado. Mejor val acc: {mejor_val_acc:.2%}")

    # Evaluación final en test
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test acc: {test_acc:.2%}")


main()
