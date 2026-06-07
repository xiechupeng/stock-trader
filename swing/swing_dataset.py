"""
Swing Token 数据集构建
从5分钟 OHLCV → ZigZag pivot → Swing token 序列 → 训练数据 (X, y)
"""
import numpy as np
import pandas as pd
from swing.zigzag import find_pivots, pivots_to_swings, zigzag_stats
from swing.swing_tokenizer import swing_to_token, encode, VOCAB_SIZE


def df_to_swing_tokens(df: pd.DataFrame,
                       threshold: float = 0.003) -> tuple[list[str], list[int]]:
    """
    单只股票 df → (token_strings, token_indices)
    session_vol_avg 用 df 全局均量
    """
    pivots = find_pivots(df, threshold=threshold)
    if len(pivots) < 2:
        return [], []

    swings = pivots_to_swings(df, pivots)
    session_vol_avg = float(df['volume'].mean())

    tokens = [swing_to_token(s, session_vol_avg) for s in swings]
    idxs   = [encode(t) for t in tokens]
    return tokens, idxs


def make_swing_sequences(token_idxs: list[int],
                         seq_len: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口切 (X, y) 对"""
    X, y = [], []
    for i in range(len(token_idxs) - seq_len):
        X.append(token_idxs[i : i + seq_len])
        y.append(token_idxs[i + seq_len])
    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


def build_dataset(dfs: dict[str, pd.DataFrame],
                  threshold: float = 0.003,
                  seq_len: int = 10,
                  train_ratio: float = 0.9) -> tuple:
    """
    多只股票 → 合并训练集
    返回 (X_tr, y_tr, X_val, y_val, token_map)
    token_map: {symbol: token_strings}
    """
    X_all, y_all = [], []
    token_map = {}

    for sym, df in dfs.items():
        tokens, idxs = df_to_swing_tokens(df, threshold)
        if len(idxs) < seq_len + 2:
            print(f'  {sym}: swing 太少 ({len(idxs)})，跳过')
            continue

        # 统计
        pivots = find_pivots(df, threshold)
        swings = pivots_to_swings(df, pivots)
        stats  = zigzag_stats(pivots, swings)
        print(f'  {sym}: {stats["n_swings"]} swings '
              f'(UP={stats["n_up"]} DN={stats["n_down"]}) '
              f'avg_mag={stats["avg_mag_%"]:.2f}% '
              f'avg_bars={stats["avg_bars"]:.1f}')

        X, y = make_swing_sequences(idxs, seq_len)
        X_all.append(X)
        y_all.append(y)
        token_map[sym] = tokens

    if not X_all:
        raise RuntimeError('所有股票 swing 数量不足，请降低 threshold 或提供更多数据')

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:], token_map
