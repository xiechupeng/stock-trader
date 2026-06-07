"""
5分钟日内数据读取层
从 cn_us_trader/data_cache/us_5min/ 读取已下载的 CSV
返回与 massive_fetcher 完全相同格式的 DataFrame（含衍生列）
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

# cn_us_trader 项目的5分钟缓存目录
DEFAULT_5MIN_DIR = "/Users/xiechupeng/cn_us_trader/data_cache/us_5min"


def load_5min_csv(symbol: str, src_dir: str = DEFAULT_5MIN_DIR) -> pd.DataFrame:
    """读取单只股票的5分钟 CSV，返回含衍生列的 DataFrame"""
    path = os.path.join(src_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"5min CSV 不存在: {path}")

    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")
    df.set_index("datetime", inplace=True)
    df.index.name = "date"
    df.index = df.index.tz_localize(None)   # 去掉 tz，与 daily 保持一致
    df.sort_index(inplace=True)

    # 衍生特征
    df["returns"]      = df["close"].pct_change()
    df["log_returns"]  = np.log(df["close"] / df["close"].shift(1))
    df["vol_avg"]      = df["volume"].rolling(78).mean()   # 78根5min = 1交易日
    df["vol_ratio"]    = df["volume"] / df["vol_avg"]
    df["range"]        = df["high"] - df["low"]
    df["body"]         = (df["close"] - df["open"]).abs()
    df["body_ratio"]   = df["body"] / df["range"].replace(0, np.nan)
    df["upper_shadow"] = df.apply(lambda r: r["high"] - max(r["open"], r["close"]), axis=1)
    df["lower_shadow"] = df.apply(lambda r: min(r["open"], r["close"]) - r["low"],  axis=1)
    df.dropna(subset=["returns"], inplace=True)
    return df


def load_5min_slice(symbol: str, start: str, end: str,
                    src_dir: str = DEFAULT_5MIN_DIR) -> pd.DataFrame:
    """按日期区间切片"""
    df = load_5min_csv(symbol, src_dir)
    return df.loc[start:end]


def load_5min_multi(symbols: list, start: str = None, end: str = None,
                    src_dir: str = DEFAULT_5MIN_DIR) -> dict[str, pd.DataFrame]:
    """批量加载多只股票"""
    result = {}
    for sym in symbols:
        try:
            df = load_5min_csv(sym, src_dir)
            if start:
                df = df.loc[start:]
            if end:
                df = df.loc[:end]
            print(f"  [5min] {sym}: {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
            result[sym] = df
        except FileNotFoundError:
            print(f"  [warn] {sym}: 5min CSV 不存在，跳过")
        except Exception as e:
            print(f"  [warn] {sym}: {e}")
    return result


# 可用的5分钟股票列表
AVAILABLE_5MIN = ["TSLA", "AMD", "NVDA", "AAPL", "HOOD", "SCHW", "SPY"]
