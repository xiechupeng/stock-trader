"""
30分钟 Swing 二分类回测
  确认 TROUGH → P(next UP swing > 1.2%) > confidence → 买入（下一根bar）
  确认 PEAK → 卖出（下一根bar）
  趋势过滤: 价格 > 26根30min MA（≈日内均线）
  止损: 默认1.5%

python backtest_swing_30m.py
python backtest_swing_30m.py --confidence 0.55 --symbols AAPL NVDA AMD
"""
import argparse, os
import numpy as np
import pandas as pd
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swing.zigzag import find_pivots, pivots_to_swings
from swing import swing_tokenizer
from backtest.metrics import calc_metrics, print_metrics
from train_swing_30m import load_30m, THRESHOLDS_30M, ZIGZAG_THRESH, CACHE_DIR_30M
from pathlib import Path


def backtest_single(df: pd.DataFrame, clf, symbol: str,
                    seq_len: int = 12,
                    min_confidence: float = 0.50,
                    stop_loss_pct: float = 0.015,   # 30min: 1.5% 止损
                    commission: float = 0.001,
                    slippage: float = 0.0005,
                    position_size_pct: float = 0.20,
                    trend_ma: int = 26) -> dict:     # 26根30min ≈ 1交易日

    pivots  = find_pivots(df, threshold=ZIGZAG_THRESH)
    swings  = pivots_to_swings(df, pivots)
    vol_avg = float(df["volume"].mean())

    tokens = [swing_tokenizer.swing_to_token(s, vol_avg, **THRESHOLDS_30M)
              for s in swings]
    idxs   = [swing_tokenizer.encode(t) for t in tokens]

    closes = df["close"].values
    dates  = df.index
    n      = len(df)
    ma     = df["close"].rolling(trend_ma, min_periods=trend_ma//2).mean().values

    capital      = 100_000.0
    position     = None
    equity_curve = []
    trades       = []

    for pi, pivot in enumerate(pivots):
        if pivot.confirm_idx < 0:
            continue

        act_bar   = min(pivot.confirm_idx + 1, n - 1)
        act_price = float(closes[act_bar])
        act_date  = dates[act_bar]

        pv = capital + (position["shares"] * float(closes[pivot.confirm_idx])
                        if position else 0)
        equity_curve.append({"date": pivot.confirm_date, "equity": pv})

        # 检查止损（在每个 pivot 确认时检查）
        if position is not None:
            cur = float(closes[min(pivot.confirm_idx, n-1)])
            if cur <= position["stop_price"]:
                slip     = cur * (1 - slippage)
                proceeds = position["shares"] * slip
                net      = proceeds * (1 - commission)
                pnl      = (net - position["cost"]) / position["cost"]
                capital += net
                trades.append({"symbol": symbol, "pnl_pct": pnl,
                               "entry_date": position["entry_date"],
                               "exit_date": act_date,
                               "entry_price": position["entry_price"],
                               "exit_price": slip, "reason": "stop_loss",
                               "hold_bars": act_bar - position["entry_bar"],
                               "prob": position["prob"]})
                position = None
                continue

        if pi < seq_len:
            continue

        ctx  = np.array([idxs[pi - seq_len : pi]], dtype=np.float32)

        # TROUGH → 趋势过滤 + 模型预测 → 买入
        if position is None and pivot.ptype == "TROUGH":
            bar_ma = ma[min(act_bar, n-1)]
            if not (np.isnan(bar_ma) or act_price >= bar_ma * 0.95):
                continue   # 低于MA，跳过

            prob = float(clf.predict_proba(ctx)[0, 1])
            if prob >= min_confidence:
                slip   = act_price * (1 + slippage)
                alloc  = capital * position_size_pct
                shares = (alloc * (1 - commission)) / slip
                capital -= alloc
                position = {
                    "entry_price": slip, "entry_date": act_date,
                    "shares": shares, "cost": alloc * (1 - commission),
                    "entry_bar": act_bar, "prob": prob,
                    "stop_price": slip * (1 - stop_loss_pct),
                }

        # PEAK → 卖出
        elif position is not None and pivot.ptype == "PEAK":
            slip     = act_price * (1 - slippage)
            proceeds = position["shares"] * slip
            net      = proceeds * (1 - commission)
            pnl      = (net - position["cost"]) / position["cost"]
            capital += net
            trades.append({"symbol": symbol, "pnl_pct": pnl,
                           "entry_date": position["entry_date"],
                           "exit_date": act_date,
                           "entry_price": position["entry_price"],
                           "exit_price": slip, "reason": "peak",
                           "hold_bars": act_bar - position["entry_bar"],
                           "prob": position["prob"]})
            position = None

    # 强制平仓
    if position and n > 0:
        last = float(closes[-1])
        net  = position["shares"] * last * (1 - slippage) * (1 - commission)
        pnl  = (net - position["cost"]) / position["cost"]
        capital += net
        trades.append({"symbol": symbol, "pnl_pct": pnl,
                       "entry_date": position["entry_date"], "exit_date": dates[-1],
                       "entry_price": position["entry_price"], "exit_price": last,
                       "reason": "end", "hold_bars": n-1-position["entry_bar"],
                       "prob": position["prob"]})

    if not equity_curve:
        return {}

    eq = pd.DataFrame(equity_curve).set_index("date")["equity"]
    m  = calc_metrics(eq, trades)
    return {"equity": eq, "trades": trades, "metrics": m, "symbol": symbol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",    nargs="+", default=None)
    parser.add_argument("--start",      default="2025-10-01")
    parser.add_argument("--end",        default="2026-06-07")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--stop_loss",  type=float, default=0.015)
    parser.add_argument("--seq_len",    type=int,   default=12)
    args = parser.parse_args()

    # 自动检测可用股票
    if args.symbols is None:
        args.symbols = sorted(f.stem for f in Path(CACHE_DIR_30M).glob("*.parquet"))
    if not args.symbols:
        print("❌ 无30min数据，请先运行: python download_30m.py")
        return

    print(f"\n{'='*54}")
    print(f"  30分钟 Swing 回测")
    print(f"  置信度: {args.confidence}  止损: {args.stop_loss*100:.1f}%")
    print(f"  时间段: {args.start} → {args.end}")
    print(f"  股票: {args.symbols}")
    print(f"{'='*54}\n")

    print("[1/3] 加载模型...")
    clf = joblib.load("saved_models/swing_30m_gbm.pkl")

    print("[2/3] 加载数据并回测...\n")
    results = {}
    buy_hold = {}

    for sym in args.symbols:
        try:
            df = load_30m(sym)
            df = df[args.start:args.end]
            if len(df) < 100:
                print(f"  {sym}: 测试期数据不足")
                continue

            bnh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
            buy_hold[sym] = bnh

            r = backtest_single(df, clf, sym,
                                seq_len=args.seq_len,
                                min_confidence=args.confidence,
                                stop_loss_pct=args.stop_loss)
            if r:
                results[sym] = r
        except Exception as e:
            print(f"  [warn] {sym}: {e}")

    if not results:
        print("无交易，尝试降低 --confidence")
        return

    # 打印结果
    print(f"\n{'='*66}")
    print(f"  {'股票':<6} {'策略':>9} {'买入持有':>9} {'超额':>8} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'胜率':>6} {'笔数':>5}")
    print(f"  {'-'*62}")
    for sym, r in sorted(results.items(), key=lambda x: -x[1]["metrics"]["total_return"]):
        m   = r["metrics"]
        bnh = buy_hold.get(sym, 0)
        alpha = m["total_return"] - bnh
        flag  = "✅" if alpha > 0 else "  "
        wr = m['win_rate'] if m['win_rate'] is not None else 0
        print(f"  {sym:<6} {m['total_return']:>+8.1f}%  {bnh:>+8.1f}%  "
              f"{alpha:>+7.1f}%  {m['sharpe']:>+6.3f}  "
              f"{m['max_drawdown']:>6.1f}%  {wr:>5.0f}%  "
              f"{m['num_trades']:>4}  {flag}")

    # Portfolio
    equities   = [v["equity"] for v in results.values()]
    all_trades = [t for v in results.values() for t in v["trades"]]
    port_eq    = pd.concat(equities, axis=1).mean(axis=1)
    port_m     = calc_metrics(port_eq, all_trades)
    port_bnh   = np.mean(list(buy_hold.values()))
    print(f"  {'='*62}")
    print(f"  {'PORT':<6} {port_m['total_return']:>+8.1f}%  {port_bnh:>+8.1f}%  "
          f"{port_m['total_return']-port_bnh:>+7.1f}%  {port_m['sharpe']:>+6.3f}  "
          f"{port_m['max_drawdown']:>6.1f}%  {port_m['win_rate']:>5.0f}%  "
          f"{port_m['num_trades']:>4}")

    # 置信度分层
    if all_trades:
        print("\n  置信度分层:")
        for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.0)]:
            sub = [t for t in all_trades if lo <= t.get("prob",0) < hi]
            if sub:
                wr  = sum(1 for t in sub if t["pnl_pct"] > 0) / len(sub)
                avg = np.mean([t["pnl_pct"] for t in sub]) * 100
                print(f"    [{lo:.2f},{hi:.2f}): n={len(sub):4d}  "
                      f"winrate={wr*100:.0f}%  avg={avg:+.3f}%")

    # 画图
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for sym, r in results.items():
        axes[0].plot(r["equity"], label=sym, alpha=0.7, lw=1)
    axes[0].plot(port_eq, color="black", lw=2.5, label="Portfolio")
    axes[0].set_title(f"30min Swing 净值  conf={args.confidence}  stop={args.stop_loss*100:.1f}%")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    dd = (port_eq - port_eq.cummax()) / port_eq.cummax() * 100
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[1].set_title("回撤 (%)"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = "results/swing_30m_backtest.png"
    plt.savefig(path, dpi=150)
    print(f"\n[plot] → {path}")


if __name__ == "__main__":
    main()
