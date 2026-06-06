"""
第二层 Token：K线形态识别
优先级从高到低：跳空 > 吞没 > 锤子/射击之星 > 大阳/大阴 > 普通涨跌 > 十字星
"""
import pandas as pd
import numpy as np
from config import TokenConfig


CANDLE_TOKENS = [
    "GAP_UP", "GAP_DOWN",
    "ENGULF_BULL", "ENGULF_BEAR",
    "HAMMER", "SHOOTING_STAR",
    "BIG_BULL", "BIG_BEAR",
    "BULL", "BEAR",
    "DOJI",
    "NORMAL",
]


def _is_gap_up(row: pd.Series, prev: pd.Series, thresh: float) -> bool:
    return row["open"] > prev["close"] * (1 + thresh)


def _is_gap_down(row: pd.Series, prev: pd.Series, thresh: float) -> bool:
    return row["open"] < prev["close"] * (1 - thresh)


def _is_engulf_bull(row: pd.Series, prev: pd.Series) -> bool:
    """前一根阴线，当前阳线完全吞没"""
    prev_bearish = prev["close"] < prev["open"]
    curr_bullish = row["close"] > row["open"]
    engulf = (row["open"] <= prev["close"]) and (row["close"] >= prev["open"])
    return prev_bearish and curr_bullish and engulf


def _is_engulf_bear(row: pd.Series, prev: pd.Series) -> bool:
    """前一根阳线，当前阴线完全吞没"""
    prev_bullish = prev["close"] > prev["open"]
    curr_bearish = row["close"] < row["open"]
    engulf = (row["open"] >= prev["close"]) and (row["close"] <= prev["open"])
    return prev_bullish and curr_bearish and engulf


def _is_hammer(row: pd.Series, cfg: TokenConfig) -> bool:
    """锤子线：下影线 > 2x 实体，上影线小，出现在下跌"""
    body = row["body"]
    if body < 1e-8:
        return False
    lower = row["lower_shadow"]
    upper = row["upper_shadow"]
    return (lower / body >= cfg.hammer_shadow_ratio) and (upper < body * 0.5)


def _is_shooting_star(row: pd.Series, cfg: TokenConfig) -> bool:
    """射击之星：上影线 > 2x 实体，下影线小，出现在上涨"""
    body = row["body"]
    if body < 1e-8:
        return False
    upper = row["upper_shadow"]
    lower = row["lower_shadow"]
    return (upper / body >= cfg.hammer_shadow_ratio) and (lower < body * 0.5)


def tokenize_candle(df: pd.DataFrame, cfg: TokenConfig) -> pd.Series:
    """
    逐行识别K线形态，返回 CANDLE_TOKENS 之一。
    df 需含列: open, high, low, close, body, body_ratio,
               upper_shadow, lower_shadow, returns
    """
    tokens = []
    rows = df.reset_index(drop=False)

    for i, row in rows.iterrows():
        if i == 0:
            tokens.append("NORMAL")
            continue

        prev = rows.iloc[i - 1]
        r = row["returns"] if not pd.isna(row["returns"]) else 0.0

        # 优先级从高到低
        if _is_gap_up(row, prev, cfg.gap_threshold):
            tokens.append("GAP_UP")
        elif _is_gap_down(row, prev, cfg.gap_threshold):
            tokens.append("GAP_DOWN")
        elif _is_engulf_bull(row, prev):
            tokens.append("ENGULF_BULL")
        elif _is_engulf_bear(row, prev):
            tokens.append("ENGULF_BEAR")
        elif _is_hammer(row, cfg) and r >= 0:
            tokens.append("HAMMER")
        elif _is_shooting_star(row, cfg) and r <= 0:
            tokens.append("SHOOTING_STAR")
        elif row["body_ratio"] >= cfg.big_body_ratio and r > 0:
            tokens.append("BIG_BULL")
        elif row["body_ratio"] >= cfg.big_body_ratio and r < 0:
            tokens.append("BIG_BEAR")
        elif row["body_ratio"] < cfg.doji_body_ratio:
            tokens.append("DOJI")
        elif r > 0:
            tokens.append("BULL")
        elif r < 0:
            tokens.append("BEAR")
        else:
            tokens.append("NORMAL")

    return pd.Series(tokens, index=df.index)
