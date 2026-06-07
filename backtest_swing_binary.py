"""
Swing 二分类回测
  在确认的 TROUGH 处：P(next UP swing > 1.5%) > threshold → 买入
  在确认的 PEAK 处：平仓

python backtest_swing_binary.py
python backtest_swing_binary.py --confidence 0.50 --mag_threshold 0.015
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.intraday_fetcher import load_5min_multi, AVAILABLE_5MIN
from swing.zigzag import find_pivots, pivots_to_swings
from swing import swing_tokenizer
from swing.binary_model import BinarySwingModel
from backtest.metrics import calc_metrics, print_metrics


def backtest_single(df: pd.DataFrame, model: BinarySwingModel, symbol: str,
                    zigzag_thresh: float = 0.003,
                    seq_len: int = 15,
                    min_confidence: float = 0.50,
                    commission: float = 0.001,
                    slippage: float = 0.0005,
                    position_size_pct: float = 0.20) -> dict:

    pivots  = find_pivots(df, threshold=zigzag_thresh)
    swings  = pivots_to_swings(df, pivots)
    vol_avg = float(df["volume"].mean())

    tokens = [swing_tokenizer.swing_to_token(s, vol_avg) for s in swings]
    idxs   = [swing_tokenizer.encode(t) for t in tokens]
    closes = df["close"].values
    dates  = df.index

    capital      = 100_000.0
    position     = None
    equity_curve = []
    trades       = []

    for pi, pivot in enumerate(pivots):
        if pivot.confirm_idx < 0:
            continue

        act_bar   = min(pivot.confirm_idx + 1, len(closes) - 1)
        act_price = float(closes[act_bar])
        act_date  = dates[act_bar]

        pv = capital
        if position:
            pv += position["shares"] * float(closes[pivot.confirm_idx])
        equity_curve.append({"date": pivot.confirm_date, "equity": pv})

        if pi < seq_len:
            continue

        context = idxs[pi - seq_len : pi]

        if position is None and pivot.ptype == "TROUGH":
            prob = model.predict_proba(context)
            if prob >= min_confidence:
                slip   = act_price * (1 + slippage)
                alloc  = capital * position_size_pct
                comm   = alloc * commission
                shares = (alloc - comm) / slip
                capital -= alloc
                position = {
                    "entry_price": slip, "entry_date": act_date,
                    "shares": shares, "cost": alloc - comm,
                    "entry_bar": act_bar, "prob": prob,
                }

        elif position is not None and pivot.ptype == "PEAK":
            slip     = act_price * (1 - slippage)
            proceeds = position["shares"] * slip
            comm     = proceeds * commission
            net      = proceeds - comm
            pnl      = (net - position["cost"]) / position["cost"]
            capital += net
            trades.append({
                "symbol": symbol,
                "entry_price": position["entry_price"],
                "exit_price":  slip,
                "entry_date":  position["entry_date"],
                "exit_date":   act_date,
                "pnl_pct":     pnl,
                "hold_bars":   act_bar - position["entry_bar"],
                "entry_prob":  position["prob"],
            })
            position = None

    # 强制平仓
    if position and len(closes) > 0:
        last = float(closes[-1])
        net  = position["shares"] * last * (1 - slippage) * (1 - commission)
        pnl  = (net - position["cost"]) / position["cost"]
        capital += net
        trades.append({
            "symbol": symbol, "entry_price": position["entry_price"],
            "exit_price": last, "entry_date": position["entry_date"],
            "exit_date": dates[-1], "pnl_pct": pnl,
            "hold_bars": len(closes) - 1 - position["entry_bar"],
            "entry_prob": position["prob"],
        })

    if not equity_curve:
        return {}

    eq = pd.DataFrame(equity_curve).set_index("date")["equity"]
    m  = calc_metrics(eq, trades)
    return {"equity": eq, "trades": trades, "metrics": m, "symbol": symbol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="+", default=AVAILABLE_5MIN)
    parser.add_argument("--zigzag_thresh", type=float, default=0.003)
    parser.add_argument("--mag_threshold", type=float, default=0.015)
    parser.add_argument("--seq_len",       type=int,   default=15)
    parser.add_argument("--confidence",    type=float, default=0.50,
                        help="P(large_swing) 阈值，越高信号越少但越精准")
    parser.add_argument("--start",         default="2025-10-01")
    parser.add_argument("--end",           default="2026-06-01")
    args = parser.parse_args()

    print(f"\n{'='*54}")
    print(f"  Swing 二分类回测")
    print(f"  大波段: UP > {args.mag_threshold*100:.1f}%  置信度: {args.confidence}")
    print(f"  时间段: {args.start} → {args.end}")
    print(f"{'='*54}\n")

    print("[1/3] 加载模型...")
    model = BinarySwingModel(vocab_size=swing_tokenizer.VOCAB_SIZE,
                             embed_dim=32, hidden_dim=128,
                             num_layers=2, dropout=0.2)
    model.load("saved_models/swing_binary.pt")

    print("[2/3] 加载数据...")
    dfs = load_5min_multi(args.symbols, start=args.start, end=args.end)

    print("[3/3] 回测...\n")
    results = {}
    for sym, df in dfs.items():
        print(f"── {sym} ──")
        r = backtest_single(df, model, sym,
                            zigzag_thresh=args.zigzag_thresh,
                            seq_len=args.seq_len,
                            min_confidence=args.confidence)
        if r:
            print_metrics(r["metrics"])
            results[sym] = r

    if not results:
        print("无交易")
        return

    equities   = [v["equity"] for v in results.values()]
    all_trades = [t for v in results.values() for t in v["trades"]]
    port_eq    = pd.concat(equities, axis=1).mean(axis=1)
    port_m     = calc_metrics(port_eq, all_trades)

    print("\n" + "="*54)
    print("  [Portfolio Summary]")
    print_metrics(port_m)

    # 分析胜率 vs 入场概率
    if all_trades:
        import numpy as np
        print("\n  入场概率分布（置信度 vs 胜率）:")
        bins = [(0.0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.0)]
        for lo, hi in bins:
            subset = [t for t in all_trades
                      if lo <= t.get("entry_prob", 0) < hi]
            if subset:
                wr = sum(1 for t in subset if t["pnl_pct"] > 0) / len(subset)
                avg = np.mean([t["pnl_pct"] for t in subset]) * 100
                print(f"    prob [{lo:.1f},{hi:.1f}): n={len(subset):3d}  "
                      f"winrate={wr*100:.0f}%  avg_pnl={avg:+.3f}%")

    # 画图
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for sym, r in results.items():
        axes[0].plot(r["equity"], label=sym, alpha=0.7, linewidth=1)
    axes[0].plot(port_eq, color="black", linewidth=2.5, label="Portfolio")
    axes[0].set_title(f"Swing 二分类净值  conf={args.confidence}  mag>{args.mag_threshold*100:.1f}%")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    dd = (port_eq - port_eq.cummax()) / port_eq.cummax() * 100
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[1].set_title("Portfolio 回撤 (%)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = "results/swing_binary_backtest.png"
    plt.savefig(path, dpi=150)
    print(f"\n[plot] → {path}")


if __name__ == "__main__":
    main()
