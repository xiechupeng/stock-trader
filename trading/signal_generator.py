"""
实盘信号生成器
给定最新 K 线数据 → 预测下一个 token → 输出 BUY / SELL / HOLD
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import TradingConfig, CFG
from tokenizer.combined import CombinedTokenizer
from data.massive_fetcher import fetch_symbol as fetch_ohlcv


@dataclass
class Signal:
    symbol:         str
    action:         str          # "BUY" | "SELL" | "HOLD"
    predicted_token: str
    confidence:     float
    timestamp:      pd.Timestamp


class SignalGenerator:

    def __init__(self, model, cfg: TradingConfig = None):
        self.model    = model
        self.cfg      = cfg or CFG.trading
        self.tokenizer = CombinedTokenizer()
        self.seq_len   = CFG.model.sequence_length

    def generate(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        df: 最近 seq_len + buffer 根 K 线（含必要列）
        返回 Signal
        """
        df_tok = self.tokenizer.tokenize(df)
        idxs   = df_tok["token_idx"].tolist()

        if len(idxs) < self.seq_len:
            return Signal(symbol, "HOLD", "<UNK>", 0.0, df_tok.index[-1])

        context = idxs[-self.seq_len:]
        pred_idx, conf = self.model.predict(context)
        pred_token     = self.tokenizer.decode(pred_idx)

        if conf >= self.cfg.min_confidence and pred_token in self.cfg.buy_tokens:
            action = "BUY"
        elif conf >= self.cfg.min_confidence and pred_token in self.cfg.sell_tokens:
            action = "SELL"
        else:
            action = "HOLD"

        return Signal(
            symbol=symbol,
            action=action,
            predicted_token=pred_token,
            confidence=conf,
            timestamp=df_tok.index[-1],
        )

    def scan_universe(self, symbols: list[str],
                      start: str = None, end: str = None) -> list[Signal]:
        """
        扫描多只股票，返回所有非 HOLD 信号
        """
        from datetime import datetime, timedelta
        end   = end   or datetime.today().strftime("%Y-%m-%d")
        start = start or (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

        signals = []
        for sym in symbols:
            try:
                df = fetch_ohlcv(sym, start, end, cache_dir=CFG.cache_dir)
                sig = self.generate(sym, df)
                if sig.action != "HOLD":
                    signals.append(sig)
                    print(f"  [{sig.action}] {sym} | token={sig.predicted_token} | "
                          f"conf={sig.confidence:.2%}")
            except Exception as e:
                print(f"  [warn] {sym}: {e}")

        return signals
