"""
数据获取层 — yfinance + 本地缓存
返回包含 OHLCV + 技术衍生列 的 DataFrame
"""
import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    下载 OHLCV 数据，优先读本地缓存。
    返回列: open, high, low, close, volume, returns, log_returns
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_{start}_{end}.parquet")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"[cache] {symbol} loaded from {cache_path}")
        return df

    print(f"[download] {symbol} {start} → {end}")
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {symbol}")

    # 统一列名小写
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    # 衍生特征
    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_avg"]     = df["volume"].rolling(20).mean()
    df["vol_ratio"]   = df["volume"] / df["vol_avg"]
    df["range"]       = df["high"] - df["low"]
    df["body"]        = (df["close"] - df["open"]).abs()
    df["body_ratio"]  = df["body"] / df["range"].replace(0, np.nan)
    df["upper_shadow"] = df.apply(
        lambda r: r["high"] - max(r["open"], r["close"]), axis=1
    )
    df["lower_shadow"] = df.apply(
        lambda r: min(r["open"], r["close"]) - r["low"], axis=1
    )

    df.dropna(inplace=True)
    df.to_parquet(cache_path)
    return df


def fetch_multi(
    symbols: list,
    start: str,
    end: str,
    cache_dir: str = "data_cache",
) -> dict[str, pd.DataFrame]:
    """批量拉取多只股票，返回 {symbol: df} 字典"""
    result = {}
    for sym in symbols:
        try:
            result[sym] = fetch_ohlcv(sym, start, end, cache_dir=cache_dir)
        except Exception as e:
            print(f"[warn] {sym} failed: {e}")
    return result
