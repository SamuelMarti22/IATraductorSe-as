import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SignLSTM(nn.Module):

    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=33, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # lee la secuencia hacia adelante y hacia atrás
            # dropout entre capas LSTM (solo aplica si num_layers > 1)
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        # bidireccional duplica el tamaño del vector oculto (adelante + atrás)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, mask):
        # x:    (batch, frames, 126)
        # mask: (batch, frames) — True en frames reales, False en padding

        # Calcular cuántos frames reales tiene cada video del batch
        lengths = mask.sum(dim=1).cpu()

        # Empaquetar la secuencia para que el LSTM ignore los frames de padding
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        _, (hidden, _) = self.lstm(packed)

        # Concatenar la última capa forward y backward del LSTM bidireccional
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_size * 2)
        out = self.dropout(out)
        out = self.fc(out)     # (batch, num_classes)

        return out
