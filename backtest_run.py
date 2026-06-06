"""
回测入口
python backtest_run.py --model markov --symbols AAPL MSFT NVDA
"""
import argparse
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from data.fetcher import fetch_multi
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
    parser.add_argument("--symbols", nargs="+", default=CFG.symbols[:5])
    parser.add_argument("--start",   default=CFG.test_start)
    parser.add_argument("--end",     default=CFG.test_end)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  回测模型: {args.model.upper()}")
    print(f"  股票池:   {args.symbols}")
    print(f"  时间段:   {args.start} → {args.end}")
    print(f"{'='*50}\n")

    print("[1/3] 加载模型...")
    model = load_model(args.model)

    print("[2/3] 拉取测试数据...")
    dfs = fetch_multi(args.symbols, args.start, args.end, cache_dir=CFG.cache_dir)

    print("[3/3] 运行回测...")
    engine  = BacktestEngine(model)
    results = engine.run_portfolio(dfs)

    plot_equity(results)
    save_results(results)
    print("\n🎉 回测完成！\n")


if __name__ == "__main__":
    main()
