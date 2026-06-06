"""
第三层 Token：成交量信号
HIGH / NORMAL / LOW
"""
import pandas as pd
from config import TokenConfig


VOLUME_TOKENS = ["HIGH", "NORMAL", "LOW"]


def tokenize_volume(vol_ratio: pd.Series, cfg: TokenConfig) -> pd.Series:
    """
    vol_ratio = volume / rolling_avg_volume
    """
    def _label(r: float) -> str:
        if pd.isna(r):       return "NORMAL"
        if r >= cfg.high_vol_ratio: return "HIGH"
        if r <= cfg.low_vol_ratio:  return "LOW"
        return "NORMAL"

    return vol_ratio.map(_label)
