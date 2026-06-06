"""
LSTM 序列预测模型
Embedding → LSTM → Dropout → Linear → softmax
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm


class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [B, seq_len]
        emb = self.embedding(x)          # [B, seq_len, embed_dim]
        out, _ = self.lstm(emb)          # [B, seq_len, hidden_dim]
        last = out[:, -1, :]             # 取最后时间步
        logits = self.fc(self.dropout(last))  # [B, vocab_size]
        return logits


class LSTMModel:
    """训练/推理封装"""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.25, lr: float = 1e-3,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps"  if torch.backends.mps.is_available() else "cpu")
        self.net = LSTMPredictor(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ── 训练 ─────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray,   y_val: np.ndarray,
            epochs: int = 100, batch_size: int = 128,
            patience: int = 12, save_path: str = "saved_models/lstm_best.pt"):

        train_ds = TensorDataset(
            torch.LongTensor(X_train), torch.LongTensor(y_train))
        val_ds   = TensorDataset(
            torch.LongTensor(X_val),   torch.LongTensor(y_val))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        best_val_loss = float("inf")
        wait = 0

        for epoch in range(1, epochs + 1):
            # ── train
            self.net.train()
            t_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                t_loss += loss.item() * len(xb)
            t_loss /= len(train_ds)

            # ── val
            v_loss, correct = 0, 0
            self.net.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.net(xb)
                    v_loss += self.criterion(logits, yb).item() * len(xb)
                    correct += (logits.argmax(1) == yb).sum().item()
            v_loss /= len(val_ds)
            v_acc = correct / len(val_ds)

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d} | train_loss={t_loss:.4f} | "
                      f"val_loss={v_loss:.4f} | val_acc={v_acc:.3f}")

            # early stopping
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                wait = 0
                self.save(save_path)
            else:
                wait += 1
                if wait >= patience:
                    print(f"  [early stop] epoch {epoch}")
                    break

        # 加载最佳权重
        self.load(save_path)
        return self

    # ── 推理 ─────────────────────────────────────────
    def predict_proba(self, context: list[int]) -> np.ndarray:
        """context: 最近 seq_len 个 token index → [vocab_size] 概率"""
        self.net.eval()
        x = torch.LongTensor([context]).to(self.device)
        with torch.no_grad():
            logits = self.net(x)[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, context: list[int]) -> tuple[int, float]:
        probs = self.predict_proba(context)
        best = int(np.argmax(probs))
        return best, float(probs[best])

    def accuracy(self, X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
        ds = TensorDataset(torch.LongTensor(X), torch.LongTensor(y))
        loader = DataLoader(ds, batch_size=batch_size)
        correct = 0
        self.net.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                correct += (self.net(xb).argmax(1) == yb).sum().item()
        return correct / len(y)

    # ── 持久化 ───────────────────────────────────────
    def save(self, path: str):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "vocab_size": self.vocab_size,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        print(f"[lstm] loaded ← {path}")
        return self
