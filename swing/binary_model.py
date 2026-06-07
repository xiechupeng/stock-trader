"""
Swing 二分类模型：只问一个问题
  "下一个 UP swing 的幅度 > threshold_pct（默认1.5%）吗？"

解决多分类时 XL token（6%占比）被忽略的问题。
使用 pos_weight 处理类不平衡。
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class BinarySwingNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)  # [B]


class BinarySwingModel:
    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, lr: float = 1e-3,
                 pos_weight: float = 4.0,
                 device: str = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        self.net = BinarySwingNet(vocab_size, embed_dim, hidden_dim,
                                  num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(self.device)
        )
        self.history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_pr": []}

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100, batch_size: int = 256,
            patience: int = 12,
            save_path: str = "saved_models/swing_binary.pt"):

        train_ds = TensorDataset(torch.LongTensor(X_tr), torch.FloatTensor(y_tr))
        val_ds   = TensorDataset(torch.LongTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        best_val_loss, wait = float("inf"), 0

        for epoch in range(1, epochs + 1):
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

            # Val
            v_loss = 0
            all_probs, all_labels = [], []
            self.net.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.net(xb)
                    v_loss += self.criterion(logits, yb).item() * len(xb)
                    all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                    all_labels.extend(yb.cpu().tolist())
            v_loss /= len(val_ds)

            # 精确率（precision@threshold=0.5）
            probs  = np.array(all_probs)
            labels = np.array(all_labels)
            preds  = (probs >= 0.5).astype(float)
            tp = ((preds == 1) & (labels == 1)).sum()
            pp = (preds == 1).sum()
            precision = tp / pp if pp > 0 else 0.0
            recall    = tp / labels.sum() if labels.sum() > 0 else 0.0

            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d} | train={t_loss:.4f} | "
                      f"val={v_loss:.4f} | prec={precision:.3f} | rec={recall:.3f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                wait = 0
                self.save(save_path)
            else:
                wait += 1
                if wait >= patience:
                    print(f"  [early stop] epoch {epoch}")
                    break

        self.load(save_path)
        return self

    def predict_proba(self, context: list[int]) -> float:
        """返回 P(next_UP_swing is XL)"""
        self.net.eval()
        x = torch.LongTensor([context]).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.net(x))[0].cpu())

    def save(self, path: str):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[binary] loaded ← {path}")
        return self
