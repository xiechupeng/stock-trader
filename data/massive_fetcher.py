"""
Massive.com S3 Flat Files 数据获取层
文档: https://massive.com/dashboard/keys (Flat Files 标签)

数据路径格式:
  us_stocks_sip/day_aggs_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz

每个文件包含当天所有股票的 OHLCV（宽表，按日期分片）
列名: ticker, volume, vwap, open, close, high, low, window_start, transactions

使用方式:
  fetcher = MassiveFetcher(access_key, secret_key)
  df = fetcher.fetch("AAPL", "2024-01-01", "2024-12-31")
"""
import os
import io
import gzip
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

BUCKET    = "flatfiles"
ENDPOINT  = "https://files.massive.com"
PREFIX    = "us_stocks_sip/day_aggs_v1"


class MassiveFetcher:

    def __init__(self, access_key: str, secret_key: str,
                 cache_dir: str = "data_cache"):
        try:
            import boto3
            from botocore.client import Config as BotoCfg
        except ImportError:
            raise ImportError("pip install boto3")

        import boto3
        from botocore.client import Config as BotoCfg
        self.s3 = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoCfg(signature_version="s3v4"),
        )
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # ── 主接口 ────────────────────────────────────────
    def fetch(self, symbol: str, start: str, end: str,
              force_refresh: bool = False) -> pd.DataFrame:
        """
        下载 symbol 在 [start, end] 区间内的日 OHLCV 数据。
        优先读本地 cache；fallback 到 yfinance 如果 S3 返回 403。
        """
        cache_path = os.path.join(
            self.cache_dir, f"{symbol}_{start}_{end}_massive.parquet"
        )
        if not force_refresh and os.path.exists(cache_path):
            print(f"[cache] {symbol} ← {cache_path}")
            return pd.read_parquet(cache_path)

        # 收集日期列表
        dates = self._date_range(start, end)
        rows = []
        failed_403 = 0

        for d in dates:
            key = f"{PREFIX}/{d.year}/{d.month:02d}/{d.strftime('%Y-%m-%d')}.csv.gz"
            try:
                obj = self.s3.get_object(Bucket=BUCKET, Key=key)
                raw = gzip.decompress(obj["Body"].read())
                day_df = pd.read_csv(io.BytesIO(raw))
                sym_row = day_df[day_df["ticker"] == symbol]
                if not sym_row.empty:
                    rows.append(sym_row.iloc[0])
            except Exception as e:
                err_str = str(e)
                if "403" in err_str or "Forbidden" in err_str:
                    failed_403 += 1
                    if failed_403 >= 3:
                        print(f"  [massive] 403 Forbidden — 账户权限不足，切换 yfinance")
                        return self._fallback_yfinance(symbol, start, end, cache_path)
                # 其他错误（周末/假日无文件）跳过

        if not rows:
            print(f"  [massive] 无数据，切换 yfinance")
            return self._fallback_yfinance(symbol, start, end, cache_path)

        df = pd.DataFrame(rows)
        df = self._normalize(df, symbol)
        df.to_parquet(cache_path)
        print(f"[massive] {symbol}: {len(df)} days saved → {cache_path}")
        return df

    def fetch_multi(self, symbols: list, start: str, end: str) -> dict:
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch(sym, start, end)
            except Exception as e:
                print(f"  [warn] {sym}: {e}")
        return result

    # ── 内部工具 ──────────────────────────────────────
    @staticmethod
    def _date_range(start: str, end: str) -> list:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end,   "%Y-%m-%d")
        dates = []
        cur = s
        while cur <= e:
            if cur.weekday() < 5:   # 跳过周末
                dates.append(cur)
            cur += timedelta(days=1)
        return dates

    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Massive 列名: ticker, open, high, low, close, volume, vwap, window_start
        统一到和 yfinance fetcher 相同的格式
        """
        # window_start 是 Unix ms → 转日期
        if "window_start" in df.columns:
            df["date"] = pd.to_datetime(df["window_start"], unit="ms").dt.date
        elif "date" not in df.columns:
            df["date"] = pd.to_datetime(df.index)
        df = df.sort_values("date").set_index("date")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # 保留 OHLCV
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        # 衍生特征（与 yfinance fetcher 完全一致）
        df["returns"]     = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["vol_avg"]     = df["volume"].rolling(20).mean()
        df["vol_ratio"]   = df["volume"] / df["vol_avg"]
        df["range"]       = df["high"] - df["low"]
        df["body"]        = (df["close"] - df["open"]).abs()
        df["body_ratio"]  = df["body"] / df["range"].replace(0, np.nan)
        df["upper_shadow"] = df.apply(
            lambda r: r["high"] - max(r["open"], r["close"]), axis=1)
        df["lower_shadow"] = df.apply(
            lambda r: min(r["open"], r["close"]) - r["low"], axis=1)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def _fallback_yfinance(symbol: str, start: str, end: str,
                           cache_path: str) -> pd.DataFrame:
        """Massive 不可用时，透明切换 yfinance"""
        from data.fetcher import fetch_ohlcv
        df = fetch_ohlcv(symbol, start, end,
                         cache_dir=os.path.dirname(cache_path))
        return df


# ─────────────────────────────────────────────────────
# 便捷工厂：根据环境变量自动选择 Massive 或 yfinance
# ─────────────────────────────────────────────────────
def get_fetcher(cache_dir: str = "data_cache"):
    """
    优先使用 Massive（如果配置了 key）；否则 yfinance。
    配置方式:
      export MASSIVE_ACCESS_KEY=f9833846-026b-4409-8dba-540ee404d06c
      export MASSIVE_SECRET_KEY=zRlCwdMyV1qB7_hQpuhdw5jhv2aq11Ie
    """
    access = os.getenv("MASSIVE_ACCESS_KEY", "")
    secret = os.getenv("MASSIVE_SECRET_KEY", "")
    if access and secret:
        print("[fetcher] 使用 Massive.com S3 数据源")
        return MassiveFetcher(access, secret, cache_dir=cache_dir)
    else:
        print("[fetcher] 使用 yfinance 数据源（未配置 MASSIVE_ACCESS_KEY）")
        return None   # 调用方用 fetch_ohlcv


def fetch_symbol(symbol: str, start: str, end: str,
                 cache_dir: str = "data_cache") -> pd.DataFrame:
    """统一接口：自动选 Massive 或 yfinance"""
    fetcher = get_fetcher(cache_dir)
    if fetcher:
        return fetcher.fetch(symbol, start, end)
    else:
        from data.fetcher import fetch_ohlcv
        return fetch_ohlcv(symbol, start, end, cache_dir=cache_dir)


def fetch_multi(symbols: list, start: str, end: str,
                cache_dir: str = "data_cache") -> dict:
    """统一接口：批量获取"""
    fetcher = get_fetcher(cache_dir)
    result = {}
    for sym in symbols:
        try:
            if fetcher:
                result[sym] = fetcher.fetch(sym, start, end)
            else:
                from data.fetcher import fetch_ohlcv
                result[sym] = fetch_ohlcv(sym, start, end, cache_dir=cache_dir)
        except Exception as e:
            print(f"  [warn] {sym}: {e}")
    return result
