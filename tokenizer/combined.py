"""
组合 Tokenizer — 三层合并为单一 token
格式: {PRICE}_{CANDLE}_{VOLUME}  例: "U2_BIG_BULL_HIGH"

同时维护 词表 (vocab): token_str → int  /  int → token_str
"""
import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional

from config import TokenConfig, CFG
from tokenizer.price_action  import tokenize_price,  PRICE_TOKENS
from tokenizer.candle_pattern import tokenize_candle, CANDLE_TOKENS
from tokenizer.volume_signal  import tokenize_volume, VOLUME_TOKENS


# ─────────────────────────────────────────────────────
# 词表构建（全量笛卡尔积 + 特殊符号）
# ─────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab() -> tuple[dict, dict]:
    """
    返回 (token2idx, idx2token)
    包含所有理论组合 + PAD + UNK
    """
    tokens = [PAD_TOKEN, UNK_TOKEN]
    for p in PRICE_TOKENS:
        for c in CANDLE_TOKENS:
            for v in VOLUME_TOKENS:
                tokens.append(f"{p}_{c}_{v}")
    token2idx = {t: i for i, t in enumerate(tokens)}
    idx2token = {i: t for t, i in token2idx.items()}
    return token2idx, idx2token


TOKEN2IDX, IDX2TOKEN = build_vocab()
VOCAB_SIZE = len(TOKEN2IDX)


# ─────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────
class CombinedTokenizer:
    """
    将 DataFrame (含 OHLCV + 衍生列) → token 序列 → int 序列
    """
    def __init__(self, cfg: TokenConfig = None):
        self.cfg = cfg or CFG.token
        self.token2idx = TOKEN2IDX
        self.idx2token = IDX2TOKEN
        self.vocab_size = VOCAB_SIZE

    def tokenize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入 df（需含 returns, vol_ratio, open/high/low/close 等列）
        输出增加列: price_token, candle_token, vol_token, token, token_idx
        """
        df = df.copy()
        df["price_token"]  = tokenize_price(df["returns"], self.cfg)
        df["candle_token"] = tokenize_candle(df, self.cfg)
        df["vol_token"]    = tokenize_volume(df["vol_ratio"], self.cfg)

        df["token"] = (
            df["price_token"] + "_" +
            df["candle_token"] + "_" +
            df["vol_token"]
        )
        df["token_idx"] = df["token"].map(
            lambda t: self.token2idx.get(t, self.token2idx[UNK_TOKEN])
        )
        return df

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        return self.idx2token.get(idx, UNK_TOKEN)

    def save_vocab(self, path: str = "saved_models/vocab.json"):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.token2idx, f, indent=2)

    @classmethod
    def load_vocab(cls, path: str = "saved_models/vocab.json") -> "CombinedTokenizer":
        obj = cls()
        with open(path) as f:
            obj.token2idx = json.load(f)
        obj.idx2token = {v: k for k, v in obj.token2idx.items()}
        obj.vocab_size = len(obj.token2idx)
        return obj


def make_sequences(token_idxs: list, seq_len: int) -> tuple:
    """
    把 token index 序列切成 (X, y) 滑窗对
    X: [B, seq_len]   y: [B]
    """
    X, y = [], []
    for i in range(len(token_idxs) - seq_len):
        X.append(token_idxs[i : i + seq_len])
        y.append(token_idxs[i + seq_len])
    import numpy as np
    return np.array(X), np.array(y)
