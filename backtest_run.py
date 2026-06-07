"""
回测入口
python backtest_run.py --model markov --symbols AAPL NVDA SPY
python backtest_run.py --model lstm --intraday       ← 5分钟数据回测
"""
import argparse
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from data.massive_fetcher import fetch_multi
from models.model_factory import load_model
from backtest.engine import BacktestEngine
from backtest.metrics import print_metrics


def plot_equity(results: dict, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── 净值曲线
    ax = axes[0]
    for sym, r in results.items():
        if sym == "__portfolio__":
            ax.plot(r["equity"], linewidth=2.5, label="Portfolio", color="black")
        else:
            ax.plot(r["equity"], linewidth=1, alpha=0.6, label=sym)
    ax.set_title("净值曲线")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 回撤
    ax2 = axes[1]
    if "__portfolio__" in results:
        eq = results["__portfolio__"]["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
        ax2.set_title("最大回撤 (%)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "backtest_equity.png")
    plt.savefig(path, dpi=150)
    print(f"[plot] 净值图保存 → {path}")


def save_results(results: dict, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    summary = {}
    all_trades = []
    for sym, r in results.items():
        if "metrics" in r:
            summary[sym] = r["metrics"]
        if "trades" in r:
            for t in r["trades"]:
                t["symbol_group"] = sym
                all_trades.append(t)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if all_trades:
        pd.DataFrame(all_trades).to_csv(
            os.path.join(save_dir, "trades.csv"), index=False
        )

    print(f"[save] 结果保存 → {save_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="lstm",
                        choices=["markov", "lstm", "transformer"])
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start",   default=CFG.test_start)
    parser.add_argument("--end",     default=CFG.test_end)
    parser.add_argument("--intraday", action="store_true",
                        help="使用5分钟数据回测（与 train --intraday 配套）")
    args = parser.parse_args()

    mode = "5分钟日内" if args.intraday else "日线"
    print(f"\n{'='*52}")
    print(f"  回测模型: {args.model.upper()}   数据: {mode}")
    print(f"  时间段:   {args.start} → {args.end}")
    print(f"{'='*52}\n")

    print("[1/3] 加载模型...")
    model = load_model(args.model)

    print("[2/3] 加载数据...")
    if args.intraday:
        from data.intraday_fetcher import load_5min_multi, AVAILABLE_5MIN
        syms = args.symbols or AVAILABLE_5MIN
        dfs  = load_5min_multi(syms, start=args.start, end=args.end)
    else:
        from data.fetcher import fetch_ohlcv
        syms = args.symbols or CFG.symbols[:5]
        dfs  = {}
        for sym in syms:
            try:
                dfs[sym] = fetch_ohlcv(sym, args.start, args.end, cache_dir=CFG.cache_dir)
            except Exception as e:
                print(f"  [warn] {sym}: {e}")

    print(f"  股票池: {list(dfs.keys())}")

    print("[3/3] 运行回测...")
    if args.intraday:
        from config import INTRADAY_TOKEN_CONFIG
        import dataclasses
        tcfg = dataclasses.replace(CFG.trading,
            stop_loss_pct   = 0.005,   # 0.5%  (5min 波动 ~0.20%)
            take_profit_pct = 0.010,   # 1.0%
            max_hold_days   = 24,      # 24根5min = 2小时
            min_confidence  = 0.15,
        )
        engine = BacktestEngine(model, cfg_trading=tcfg)
        # 5min 回测用 intraday tokenizer
        from tokenizer.combined import CombinedTokenizer
        engine.tokenizer = CombinedTokenizer(cfg=INTRADAY_TOKEN_CONFIG)
    else:
        engine = BacktestEngine(model)
    results = engine.run_portfolio(dfs)

    plot_equity(results)
    save_results(results)
    print("\n🎉 回测完成！\n")


if __name__ == "__main__":
    main()
