"""
Massive.com REST API 数据获取层
端点: https://api.massive.com/v2/aggs/ticker/{ticker}/range/{mult}/{timespan}/{from}/{to}
认证: Authorization: Bearer {api_key}
限速: 免费版 5次/分钟，历史2年

用法:
    from data.massive_fetcher import fetch_symbol, fetch_multi
    df = fetch_symbol("AAPL", "2024-01-01", "2024-12-31")
"""
import os
import time
import warnings
import pandas as pd
import numpy as np
import requests
from pathlib import Path

warnings.filterwarnings("ignore")

MASSIVE_API_BASE = "https://api.massive.com/v2/aggs/ticker"
_RATE_LIMIT_SLEEP = 12   # 免费版 5次/分钟 → 每次等12秒


# ─────────────────────────────────────────────────────
# 核心 API 调用
# ─────────────────────────────────────────────────────
def _call_api(ticker: str, multiplier: int, timespan: str,
              from_date: str, to_date: str, api_key: str) -> list[dict]:
    """
    调用 Massive REST API，返回 results 列表。
    timespan: "day" | "minute" | "hour"
    """
    url = f"{MASSIVE_API_BASE}/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params  = {"adjusted": "true", "limit": 50000}

    all_results = []
    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code == 429:
            print(f"  [rate limit] 等待60秒...")
            time.sleep(60)
            continue

        if resp.status_code == 403:
            msg = resp.json().get("message", "")
            raise PermissionError(f"403 Forbidden: {msg}")

        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data.get("results", []))

        # 分页：如果有 next_url 则继续
        url    = data.get("next_url")
        params = {}  # next_url 已包含所有参数

    return all_results


# ─────────────────────────────────────────────────────
# 日线数据
# ─────────────────────────────────────────────────────
def _results_to_df(results: list[dict]) -> pd.DataFrame:
    """把 Massive results 列表转成标准 OHLCV DataFrame"""
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    # 列名映射：o/h/l/c/v/vw/t/n → open/high/low/close/volume/vwap/timestamp/trades
    rename = {"o": "open", "h": "high", "l": "low", "c": "close",
               "v": "volume", "vw": "vwap", "t": "timestamp", "n": "trades"}
    df.rename(columns=rename, inplace=True)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC") \
                   .dt.tz_convert("America/New_York").dt.normalize()
    df.set_index("date", inplace=True)
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df.sort_index(inplace=True)

    # 衍生特征（backtest engine 需要）
    df["returns"]      = df["close"].pct_change()
    df["log_returns"]  = np.log(df["close"] / df["close"].shift(1))
    df["vol_avg"]      = df["volume"].rolling(20).mean()
    df["vol_ratio"]    = df["volume"] / df["vol_avg"]
    df["range"]        = df["high"] - df["low"]
    df["body"]         = (df["close"] - df["open"]).abs()
    df["body_ratio"]   = df["body"] / df["range"].replace(0, np.nan)
    df["upper_shadow"] = df.apply(lambda r: r["high"] - max(r["open"], r["close"]), axis=1)
    df["lower_shadow"] = df.apply(lambda r: min(r["open"], r["close"]) - r["low"],  axis=1)
    df.dropna(subset=["returns"], inplace=True)
    return df


# ─────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────
def fetch_symbol(symbol: str, start: str, end: str,
                 timespan: str = "day", multiplier: int = 1,
                 cache_dir: str = "data_cache",
                 api_key: str = None,
                 force_refresh: bool = False) -> pd.DataFrame:
    """
    下载单只股票数据，优先读本地 parquet 缓存。
    api_key 优先级: 参数 > 环境变量 MASSIVE_API_KEY > 默认内置
    """
    api_key = api_key or os.getenv("MASSIVE_API_KEY", "zRlCwdMyV1qB7_hQpuhdw5jhv2aq11Ie")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_{start}_{end}_{multiplier}{timespan}.parquet")

    if not force_refresh and os.path.exists(cache_path):
        print(f"[cache] {symbol} ← {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[massive] {symbol} {multiplier}/{timespan} {start} → {end}")
    try:
        results = _call_api(symbol, multiplier, timespan, start, end, api_key)
        if not results:
            raise ValueError(f"空数据: {symbol} {start}~{end}")
        df = _results_to_df(results)
        df.to_parquet(cache_path)
        print(f"  → {len(df)} bars 已缓存")
        time.sleep(_RATE_LIMIT_SLEEP)   # 限速保护
        return df

    except PermissionError as e:
        print(f"  [warn] {e} → 降级 yfinance")
        return _fallback_yfinance(symbol, start, end, cache_path)
    except Exception as e:
        print(f"  [warn] {symbol} API 失败: {e} → 降级 yfinance")
        return _fallback_yfinance(symbol, start, end, cache_path)


def fetch_multi(symbols: list, start: str, end: str,
                timespan: str = "day", multiplier: int = 1,
                cache_dir: str = "data_cache",
                api_key: str = None) -> dict[str, pd.DataFrame]:
    """批量获取，自动限速"""
    result = {}
    for i, sym in enumerate(symbols):
        try:
            result[sym] = fetch_symbol(sym, start, end, timespan, multiplier,
                                       cache_dir=cache_dir, api_key=api_key)
        except Exception as e:
            print(f"  [warn] {sym}: {e}")
    return result


# ─────────────────────────────────────────────────────
# yfinance 降级
# ─────────────────────────────────────────────────────
def _fallback_yfinance(symbol: str, start: str, end: str, cache_path: str) -> pd.DataFrame:
    from data.fetcher import fetch_ohlcv
    df = fetch_ohlcv(symbol, start, end, cache_dir=os.path.dirname(cache_path))
    return df
