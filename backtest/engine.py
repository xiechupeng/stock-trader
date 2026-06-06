"""
回测引擎 — 事件驱动，逐日模拟
支持：止损、止盈、强制平仓、多标的同时持仓
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from config import TradingConfig, BacktestConfig, CFG
from tokenizer.combined import CombinedTokenizer, make_sequences
from backtest.metrics import calc_metrics, print_metrics


# ─────────────────────────────────────────────────────
# 持仓记录
# ─────────────────────────────────────────────────────
@dataclass
class Position:
    symbol:       str
    side:         str          # "long" | "short"
    entry_date:   pd.Timestamp
    entry_price:  float
    shares:       float
    cost:         float        # 含手续费
    stop_loss:    float
    take_profit:  float
    hold_days:    int = 0


# ─────────────────────────────────────────────────────
# 回测引擎
# ─────────────────────────────────────────────────────
class BacktestEngine:

    def __init__(self, model, cfg_trading: TradingConfig = None,
                 cfg_bt: BacktestConfig = None):
        self.model   = model
        self.tcfg    = cfg_trading or CFG.trading
        self.btcfg   = cfg_bt     or CFG.backtest
        self.tokenizer = CombinedTokenizer()
        self.seq_len   = CFG.model.sequence_length

    # ── 单只股票回测 ──────────────────────────────────
    def run_single(self, df: pd.DataFrame, symbol: str = "STOCK") -> dict:
        """
        df: tokenize 之后的 DataFrame（含 token_idx, close 列）
        """
        df = self.tokenizer.tokenize(df)
        token_idxs = df["token_idx"].tolist()
        dates      = df.index.tolist()
        closes     = df["close"].tolist()

        capital  = self.btcfg.initial_capital
        position: Optional[Position] = None
        equity_curve = []
        trades = []

        for i in range(self.seq_len, len(df)):
            date  = dates[i]
            price = closes[i]
            ctx   = token_idxs[i - self.seq_len : i]

            # ── 当前资产估值
            port_value = capital
            if position:
                if position.side == "long":
                    port_value += position.shares * price
                else:
                    pnl = position.shares * (position.entry_price - price)
                    port_value += pnl + position.cost  # 还回本金
            equity_curve.append({"date": date, "equity": port_value})

            # ── 已有仓位：检查止损/止盈/强制平仓
            if position:
                position.hold_days += 1
                exit_reason = None

                if position.side == "long":
                    if price <= position.stop_loss:    exit_reason = "stop_loss"
                    elif price >= position.take_profit: exit_reason = "take_profit"
                else:
                    if price >= position.stop_loss:    exit_reason = "stop_loss"
                    elif price <= position.take_profit: exit_reason = "take_profit"

                if position.hold_days >= self.tcfg.max_hold_days:
                    exit_reason = "max_hold"

                if exit_reason:
                    capital, trade_rec = self._close_position(position, price, exit_reason)
                    trades.append(trade_rec)
                    position = None
                    continue

            # ── 无仓位：预测下一个 token，决定开仓
            if position is None:
                pred_idx, conf = self.model.predict(ctx)
                pred_token     = self.tokenizer.decode(pred_idx)

                if conf >= self.tcfg.min_confidence:
                    if pred_token in self.tcfg.buy_tokens:
                        position, capital = self._open_position(
                            symbol, "long", date, price, capital
                        )
                    elif pred_token in self.tcfg.sell_tokens:
                        position, capital = self._open_position(
                            symbol, "short", date, price, capital
                        )

        # ── 强制平仓剩余仓位
        if position and closes:
            capital, trade_rec = self._close_position(position, closes[-1], "end_of_test")
            trades.append(trade_rec)

        equity_df = pd.DataFrame(equity_curve).set_index("date")["equity"]
        metrics   = calc_metrics(equity_df, trades)
        return {
            "symbol":  symbol,
            "equity":  equity_df,
            "trades":  trades,
            "metrics": metrics,
        }

    # ── 多标的回测（独立资金池）────────────────────────
    def run_portfolio(self, dfs: dict[str, pd.DataFrame]) -> dict:
        results = {}
        for sym, df in dfs.items():
            print(f"\n[backtest] {sym} ...")
            try:
                r = self.run_single(df, sym)
                results[sym] = r
                print_metrics(r["metrics"])
            except Exception as e:
                print(f"  [warn] {sym} error: {e}")

        # 合并净值（等权平均）
        equities = [v["equity"] for v in results.values() if "equity" in v]
        if equities:
            combined = pd.concat(equities, axis=1).mean(axis=1)
            combined.name = "portfolio"
            all_trades = [t for v in results.values() for t in v.get("trades", [])]
            metrics = calc_metrics(combined, all_trades)
            print("\n[Portfolio Summary]")
            print_metrics(metrics)
            results["__portfolio__"] = {
                "equity":  combined,
                "trades":  all_trades,
                "metrics": metrics,
            }
        return results

    # ── 内部方法 ──────────────────────────────────────
    def _open_position(self, symbol, side, date, price, capital) -> tuple[Position, float]:
        size_pct = self.tcfg.position_size_pct
        alloc    = capital * size_pct
        comm     = alloc * self.btcfg.commission
        slip     = price * (1 + self.btcfg.slippage if side == "long" else 1 - self.btcfg.slippage)
        shares   = (alloc - comm) / slip

        if side == "long":
            sl = price * (1 - self.tcfg.stop_loss_pct)
            tp = price * (1 + self.tcfg.take_profit_pct)
        else:
            sl = price * (1 + self.tcfg.stop_loss_pct)
            tp = price * (1 - self.tcfg.take_profit_pct)

        pos = Position(
            symbol=symbol, side=side, entry_date=date,
            entry_price=price, shares=shares,
            cost=alloc - comm,
            stop_loss=sl, take_profit=tp,
        )
        capital -= alloc
        return pos, capital

    def _close_position(self, pos: Position, price: float, reason: str) -> tuple[float, dict]:
        slip   = price * (1 - self.btcfg.slippage if pos.side == "long" else 1 + self.btcfg.slippage)
        if pos.side == "long":
            proceeds = pos.shares * slip
        else:
            proceeds = pos.cost + pos.shares * (pos.entry_price - slip)

        comm     = proceeds * self.btcfg.commission
        net      = proceeds - comm
        pnl_pct  = (net - pos.cost) / pos.cost

        trade_rec = {
            "symbol":      pos.symbol,
            "side":        pos.side,
            "entry_date":  pos.entry_date,
            "exit_date":   None,          # 由调用方填写（此处略）
            "entry_price": pos.entry_price,
            "exit_price":  price,
            "pnl_pct":     pnl_pct,
            "hold_days":   pos.hold_days,
            "reason":      reason,
        }
        return net, trade_rec


# ─────────────────────────────────────────────────────
# 便捷函数：一行启动回测
# ─────────────────────────────────────────────────────
def run_backtest(model, dfs: dict[str, pd.DataFrame]) -> dict:
    engine = BacktestEngine(model)
    return engine.run_portfolio(dfs)
