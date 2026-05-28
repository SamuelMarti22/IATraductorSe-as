import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataloader import get_dataloaders
from model import SignLSTM


BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3
HIDDEN_SIZE = 384
NUM_LAYERS  = 2
DROPOUT     = 0.3


def plot_metrics(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].plot(epochs, history["test_loss"], label="Test loss")
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train acc")
    axes[1].plot(epochs, history["val_acc"], label="Val acc")
    axes[1].plot(epochs, history["test_acc"], label="Test acc")
    axes[1].set_title("Accuracy por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_with_confusion_matrix(model, loader, num_classes, device):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for secuencias, mascara, labels in loader:
            secuencias = secuencias.to(device)
            mascara = mascara.to(device)
            labels = labels.to(device)

            outputs = model(secuencias, mascara)
            preds = outputs.argmax(dim=1)

            for real, pred in zip(labels.view(-1), preds.view(-1)):
                confusion[real.item(), pred.item()] += 1

    precisions = []
    recalls = []
    f1_scores = []

    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return confusion, np.array(precisions), np.array(recalls), np.array(f1_scores)


def plot_f1_scores(f1_scores, class_names, output_path):
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.5), 5))
    bars = ax.bar(class_names, f1_scores, color="#4C78A8")

    ax.set_ylim(0, 1)
    ax.set_title("F1-score por clase")
    ax.set_xlabel("Clase")
    ax.set_ylabel("F1-score")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    graphics_dir = base / "graphics"
    graphics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")

    train_loader, val_loader, test_loader, label_to_idx = get_dataloaders(csv_path, BATCH_SIZE)
    num_classes = len(label_to_idx)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(num_classes)]
    print(f"Clases: {num_classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model     = SignLSTM(input_size=126, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                         num_classes=num_classes, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Reduce el learning rate a la mitad si val_loss no mejora en 5 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mejor_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    plot_path = graphics_dir / "training_metrics.png"
    f1_plot_path = graphics_dir / "f1_por_clase.png"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader,   criterion, device)
        # Se evalúa en test solo para visualizar la evolución; no se usa para elegir hiperparámetros.
        test_loss,  test_acc  = eval_epoch(model, test_loader,  criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.2%} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.2%} | "
              f"Test loss: {test_loss:.4f} acc: {test_acc:.2%}")

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

    confusion, precisions, recalls, f1_scores = evaluate_with_confusion_matrix(
        model, test_loader, num_classes, device
    )

    print("\nMétricas por clase en test:")
    print(f"{'Clase':<20} {'Precisión':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 82)
    for class_idx, class_name in enumerate(class_names):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp
        print(
            f"{class_name:<20} {precisions[class_idx]:>9.2%} {recalls[class_idx]:>9.2%} "
            f"{f1_scores[class_idx]:>9.2%} {tp:>8} {fp:>8} {fn:>8}"
        )

    plot_metrics(history, plot_path)
    print(f"Gráfica guardada en: {plot_path}")

    plot_f1_scores(f1_scores, class_names, f1_plot_path)
    print(f"Gráfica F1 guardada en: {f1_plot_path}")


main()
