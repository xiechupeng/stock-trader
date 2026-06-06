"""
GPT-style Causal Transformer 模型
Embedding + Positional Encoding → TransformerEncoder → Linear → softmax
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# ─────────────────────────────────────────────────────
# 位置编码
# ─────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────
# Transformer 网络
# ─────────────────────────────────────────────────────
class TransformerPredictor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
                 seq_len: int = 30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN 更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Causal mask（只看当前及之前位置）
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(seq_len)
        )

        self.fc = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # x: [B, seq_len]
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len].to(x.device)
        emb  = self.pos_enc(self.embedding(x))          # [B, seq, d_model]
        out  = self.transformer(emb, mask=mask,
                                is_causal=True)         # [B, seq, d_model]
        logits = self.fc(out[:, -1, :])                 # 取最后位置
        return logits


# ─────────────────────────────────────────────────────
# 训练封装（与 LSTMModel 接口一致）
# ─────────────────────────────────────────────────────
class TransformerModel:
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
                 seq_len: int = 30, lr: float = 1e-3, device: str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps"  if torch.backends.mps.is_available() else "cpu")
        self.net = TransformerPredictor(
            vocab_size, d_model, nhead, num_layers, dim_ff, dropout, seq_len
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=128, patience=12,
            save_path="saved_models/transformer_best.pt"):

        train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
        val_ds   = TensorDataset(torch.LongTensor(X_val),   torch.LongTensor(y_val))
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
            self.scheduler.step()
            t_loss /= len(train_ds)

            v_loss, correct = 0, 0
            self.net.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.net(xb)
                    v_loss  += self.criterion(logits, yb).item() * len(xb)
                    correct += (logits.argmax(1) == yb).sum().item()
            v_loss /= len(val_ds)
            v_acc = correct / len(val_ds)

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            if epoch % 10 == 0 or epoch == 1:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(f"  epoch {epoch:3d} | train={t_loss:.4f} | "
                      f"val={v_loss:.4f} | acc={v_acc:.3f} | lr={lr_now:.2e}")

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

    def predict_proba(self, context: list[int]) -> np.ndarray:
        self.net.eval()
        x = torch.LongTensor([context]).to(self.device)
        with torch.no_grad():
            logits = self.net(x)[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, context: list[int]) -> tuple[int, float]:
        probs = self.predict_proba(context)
        best = int(np.argmax(probs))
        return best, float(probs[best])

    def accuracy(self, X, y, batch_size=512) -> float:
        ds = TensorDataset(torch.LongTensor(X), torch.LongTensor(y))
        loader = DataLoader(ds, batch_size=batch_size)
        correct = 0
        self.net.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                correct += (self.net(xb).argmax(1) == yb).sum().item()
        return correct / len(y)

    def save(self, path: str):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "vocab_size": self.vocab_size}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        print(f"[transformer] loaded ← {path}")
        return self
