"""
从 Massive API 下载 30 分钟 OHLCV 数据
保存到 data_cache/30min/{SYMBOL}.parquet

python download_30m.py
python download_30m.py --symbols AAPL AMD NVDA --start 2024-01-01
"""
import argparse
import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

API_KEY   = "zRlCwdMyV1qB7_hQpuhdw5jhv2aq11Ie"
BASE_URL  = "https://api.massive.com/v2/aggs/ticker"
CACHE_DIR = "data_cache/30min"

SYMBOLS_30M = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "AMD", "SPY", "QQQ",
    "JPM", "NFLX", "HOOD", "SCHW", "V",
]


def fetch_30min(symbol: str, start: str, end: str) -> pd.DataFrame:
    """下载单只股票的 30 分钟数据（自动分页）"""
    url    = f"{BASE_URL}/{symbol}/range/30/minute/{start}/{end}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params  = {"adjusted": "true", "limit": 50000}

    all_results = []
    page = 1
    while url:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 429:
            print(f"    ⚠️  429限速，等待65s...")
            time.sleep(65)
            continue
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        all_results.extend(results)
        print(f"    第{page}页: +{len(results):,} 条  累计={len(all_results):,}")
        url    = data.get("next_url")
        params = {}
        page  += 1
        if url:
            time.sleep(12)   # 5次/分钟限速

    if not all_results:
        raise ValueError(f"无数据: {symbol}")

    df = pd.DataFrame(all_results)
    df["datetime"] = (pd.to_datetime(df["t"], unit="ms")
                       .dt.tz_localize("UTC")
                       .dt.tz_convert("America/New_York")
                       .dt.tz_localize(None))
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)

    # 衍生特征
    df["returns"]      = df["close"].pct_change()
    df["log_returns"]  = np.log(df["close"] / df["close"].shift(1))
    df["vol_avg"]      = df["volume"].rolling(26).mean()   # 26根30min≈1交易日
    df["vol_ratio"]    = df["volume"] / df["vol_avg"]
    df["range"]        = df["high"] - df["low"]
    df["body"]         = (df["close"] - df["open"]).abs()
    df["body_ratio"]   = df["body"] / df["range"].replace(0, np.nan)
    df["upper_shadow"] = df.apply(lambda r: r["high"]-max(r["open"],r["close"]), axis=1)
    df["lower_shadow"] = df.apply(lambda r: min(r["open"],r["close"])-r["low"], axis=1)
    df.dropna(subset=["returns"], inplace=True)
    df.index.name = "date"
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS_30M)
    parser.add_argument("--start",   default="2024-06-01")
    parser.add_argument("--end",     default="2026-06-07")
    args = parser.parse_args()

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n下载 30分钟数据: {args.start} → {args.end}")
    print(f"股票: {args.symbols}\n")

    for i, sym in enumerate(args.symbols, 1):
        path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"[{i}/{len(args.symbols)}] {sym}: 已有缓存 ({len(df):,} bars)")
            continue

        print(f"[{i}/{len(args.symbols)}] {sym}: 下载中...")
        try:
            df = fetch_30min(sym, args.start, args.end)
            df.to_parquet(path)
            print(f"  ✓ {len(df):,} bars 保存 → {path}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")

        if i < len(args.symbols):
            print(f"  ⏳ 等待13s...")
            time.sleep(13)

    print("\n✅ 下载完成！")


if __name__ == "__main__":
    main()
