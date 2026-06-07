"""
日线 Swing 二分类回测
  确认 TROUGH → P(next UP swing > 5%) > confidence → 买入（次日开盘）
  确认 PEAK → 卖出（次日开盘）

python backtest_swing_daily.py
python backtest_swing_daily.py --confidence 0.55 --symbols AAPL NVDA TSLA SPY
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.fetcher import fetch_ohlcv
from swing.zigzag import find_pivots, pivots_to_swings
from swing import swing_tokenizer
from swing.binary_model import BinarySwingModel
from backtest.metrics import calc_metrics, print_metrics
from train_swing_daily import SYMBOLS_DAILY, DAILY_ZIGZAG_THRESH


def backtest_single(df: pd.DataFrame, model: BinarySwingModel, symbol: str,
                    zigzag_thresh: float = 0.03,
                    seq_len: int = 12,
                    min_confidence: float = 0.50,
                    commission: float = 0.001,
                    slippage: float = 0.0005,
                    position_size_pct: float = 0.20) -> dict:
    """
    日线 Swing 回测
    入场：TROUGH 确认当天的【次日 open】（模拟实盘：收盘后确认，次日开盘下单）
    出场：PEAK 确认当天的【次日 open】
    """
    pivots  = find_pivots(df, threshold=zigzag_thresh)
    swings  = pivots_to_swings(df, pivots)
    vol_avg = float(df["volume"].mean())

    tokens = [swing_tokenizer.swing_to_token(
                  s, vol_avg,
                  mag_thresh=swing_tokenizer.DAILY_MAG_THRESHOLDS,
                  dur_thresh=swing_tokenizer.DAILY_DUR_THRESHOLDS)
              for s in swings]
    idxs   = [swing_tokenizer.encode(t) for t in tokens]

    opens  = df["open"].values
    closes = df["close"].values
    dates  = df.index
    n      = len(df)

    capital      = 100_000.0
    position     = None
    equity_curve = []
    trades       = []

    for pi, pivot in enumerate(pivots):
        if pivot.confirm_idx < 0:
            continue

        # 日线实盘：确认当天是 T，次日 T+1 开盘下单
        trade_bar = min(pivot.confirm_idx + 1, n - 1)
        trade_price = float(opens[trade_bar])   # 用 open，模拟开盘下单
        trade_date  = dates[trade_bar]

        # 净值（以确认日收盘估值）
        pv = capital
        if position:
            pv += position["shares"] * float(closes[pivot.confirm_idx])
        equity_curve.append({"date": pivot.confirm_date, "equity": pv})

        if pi < seq_len:
            continue

        context = idxs[pi - seq_len : pi]

        # ── TROUGH 确认 → 预测 → 买入
        if position is None and pivot.ptype == "TROUGH":
            prob = model.predict_proba(context)
            if prob >= min_confidence:
                slip   = trade_price * (1 + slippage)
                alloc  = capital * position_size_pct
                comm   = alloc * commission
                shares = (alloc - comm) / slip
                capital -= alloc
                position = {
                    "entry_price": slip, "entry_date": trade_date,
                    "shares": shares,    "cost": alloc - comm,
                    "entry_bar": trade_bar, "prob": prob,
                }

        # ── PEAK 确认 → 次日开盘卖出
        elif position is not None and pivot.ptype == "PEAK":
            slip     = trade_price * (1 - slippage)
            proceeds = position["shares"] * slip
            comm     = proceeds * commission
            net      = proceeds - comm
            pnl      = (net - position["cost"]) / position["cost"]
            capital += net
            trades.append({
                "symbol":      symbol,
                "entry_price": position["entry_price"],
                "exit_price":  slip,
                "entry_date":  position["entry_date"],
                "exit_date":   trade_date,
                "pnl_pct":     pnl,
                "hold_days":   trade_bar - position["entry_bar"],
                "entry_prob":  position["prob"],
            })
            position = None

    # 强制平仓
    if position and n > 0:
        last = float(closes[-1])
        net  = position["shares"] * last * (1 - slippage) * (1 - commission)
        pnl  = (net - position["cost"]) / position["cost"]
        capital += net
        trades.append({
            "symbol": symbol, "pnl_pct": pnl,
            "entry_date": position["entry_date"], "exit_date": dates[-1],
            "entry_price": position["entry_price"], "exit_price": last,
            "hold_days": n - 1 - position["entry_bar"],
            "entry_prob": position["prob"],
        })

    if not equity_curve:
        return {}

    eq = pd.DataFrame(equity_curve).set_index("date")["equity"]
    m  = calc_metrics(eq, trades)
    return {"equity": eq, "trades": trades, "metrics": m, "symbol": symbol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="+", default=SYMBOLS_DAILY)
    parser.add_argument("--start",         default="2025-01-01")
    parser.add_argument("--end",           default="2026-06-07")
    parser.add_argument("--zigzag_thresh", type=float, default=DAILY_ZIGZAG_THRESH)
    parser.add_argument("--seq_len",       type=int,   default=12)
    parser.add_argument("--confidence",    type=float, default=0.55)
    args = parser.parse_args()

    print(f"\n{'='*56}")
    print(f"  日线 Swing 回测")
    print(f"  ZigZag threshold : {args.zigzag_thresh*100:.0f}%")
    print(f"  置信度           : {args.confidence}")
    print(f"  时间段           : {args.start} → {args.end}")
    print(f"{'='*56}\n")

    print("[1/3] 加载模型...")
    model = BinarySwingModel(vocab_size=swing_tokenizer.VOCAB_SIZE,
                             embed_dim=32, hidden_dim=256,
                             num_layers=2, dropout=0.25)
    model.load("saved_models/swing_daily_binary.pt")

    print("[2/3] 加载日线数据...")
    results = {}
    for sym in args.symbols:
        try:
            df = fetch_ohlcv(sym, args.start, args.end, cache_dir="data_cache")
            print(f"  {sym}: {len(df)} days")

            print(f"  回测 {sym}...")
            r = backtest_single(df, model, sym,
                                zigzag_thresh=args.zigzag_thresh,
                                seq_len=args.seq_len,
                                min_confidence=args.confidence)
            if r and r["trades"]:
                results[sym] = r
        except Exception as e:
            print(f"  [warn] {sym}: {e}")

    if not results:
        print("无交易，尝试降低 --confidence")
        return

    # 打印结果
    print("\n" + "="*56)
    for sym, r in results.items():
        m = r["metrics"]
        print(f"  {sym:<6} | {m['total_return']:+.2f}% | "
              f"Sharpe={m['sharpe']:+.3f} | MaxDD={m['max_drawdown']:.2f}% | "
              f"WinRate={m['win_rate']}% | Trades={m['num_trades']}")

    equities   = [v["equity"] for v in results.values()]
    all_trades = [t for v in results.values() for t in v["trades"]]
    port_eq    = pd.concat(equities, axis=1).mean(axis=1)
    port_m     = calc_metrics(port_eq, all_trades)

    print("="*56)
    print(f"  Portfolio | {port_m['total_return']:+.2f}% | "
          f"Sharpe={port_m['sharpe']:+.3f} | MaxDD={port_m['max_drawdown']:.2f}% | "
          f"WinRate={port_m['win_rate']}% | Trades={port_m['num_trades']}")
    print("="*56)
    print_metrics(port_m)

    # 入场概率 vs 胜率分析
    if all_trades:
        print("  置信度分层分析:")
        for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.0)]:
            sub = [t for t in all_trades if lo <= t.get("entry_prob",0) < hi]
            if sub:
                wr  = sum(1 for t in sub if t["pnl_pct"] > 0) / len(sub)
                avg = np.mean([t["pnl_pct"] for t in sub]) * 100
                print(f"    [{lo:.2f},{hi:.2f}): n={len(sub):3d}  "
                      f"winrate={wr*100:.0f}%  avg={avg:+.2f}%")

    # 画图
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for sym, r in results.items():
        axes[0].plot(r["equity"], label=sym, alpha=0.7, linewidth=1)
    axes[0].plot(port_eq, color="black", linewidth=2.5, label="Portfolio")
    axes[0].set_title(
        f"日线 Swing 净值  ZigZag={args.zigzag_thresh*100:.0f}%  conf={args.confidence}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    dd = (port_eq - port_eq.cummax()) / port_eq.cummax() * 100
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[1].set_title("Portfolio 回撤 (%)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = "results/swing_daily_backtest.png"
    plt.savefig(path, dpi=150)
    print(f"\n[plot] → {path}")


if __name__ == "__main__":
    main()
