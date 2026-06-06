"""
第一层 Token：价格动作离散化
输入：日收益率序列
输出：U2 / U1 / N / D1 / D2
"""
import pandas as pd
import numpy as np
from config import TokenConfig


PRICE_TOKENS = ["U2", "U1", "N", "D1", "D2"]


def tokenize_price(returns: pd.Series, cfg: TokenConfig) -> pd.Series:
    """
    将日收益率映射为价格 token。
    """
    def _label(r: float) -> str:
        if r >= cfg.strong_up:   return "U2"
        elif r >= cfg.weak_up:   return "U1"
        elif r <= cfg.strong_down: return "D2"
        elif r <= cfg.weak_down:   return "D1"
        else:                    return "N"

    return returns.map(_label)
