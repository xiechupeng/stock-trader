"""
回测绩效指标
输入: 每日组合净值序列 (pd.Series)
输出: 字典 {指标名: 值}
"""
import numpy as np
import pandas as pd


def calc_metrics(equity: pd.Series, trades: list[dict] = None,
                 risk_free_rate: float = 0.05) -> dict:
    """
    equity: 每日净值, index=日期
    trades: [{"entry_price", "exit_price", "side"("long"/"short")}, ...]
    """
    ret = equity.pct_change().dropna()
    ann = 252

    # ── 基础 ──────────────────────────────────────────
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    days = (equity.index[-1] - equity.index[0]).days
    cagr = (1 + total_return) ** (365 / max(days, 1)) - 1

    # ── 波动率 / Sharpe / Sortino ─────────────────────
    daily_rf = risk_free_rate / ann
    excess   = ret - daily_rf
    vol_ann  = ret.std() * np.sqrt(ann)

    sharpe  = (excess.mean() / (ret.std() + 1e-12)) * np.sqrt(ann)

    downside = ret[ret < 0].std() * np.sqrt(ann)
    sortino  = (ret.mean() * ann - risk_free_rate) / (downside + 1e-12)

    # ── 最大回撤 ──────────────────────────────────────
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd   = drawdown.min()

    calmar   = cagr / abs(max_dd + 1e-12)

    # ── 胜率 / 盈亏比 ─────────────────────────────────
    win_rate, profit_factor = None, None
    if trades:
        pnls = []
        for t in trades:
            ep, xp = t["entry_price"], t["exit_price"]
            pnl = (xp - ep) / ep if t.get("side", "long") == "long" else (ep - xp) / ep
            pnls.append(pnl)
        pnls = np.array(pnls)
        wins  = pnls[pnls > 0]
        loses = pnls[pnls <= 0]
        win_rate      = len(wins) / len(pnls) if len(pnls) else 0
        profit_factor = (wins.sum() / abs(loses.sum() + 1e-12)) if len(loses) else float("inf")

    result = {
        "total_return":  round(total_return * 100, 2),   # %
        "cagr":          round(cagr * 100, 2),
        "vol_ann":       round(vol_ann * 100, 2),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "max_drawdown":  round(max_dd * 100, 2),
        "calmar":        round(calmar, 3),
        "win_rate":      round(win_rate * 100, 2) if win_rate is not None else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "num_trades":    len(trades) if trades else 0,
    }
    return result


def print_metrics(m: dict):
    print("\n" + "="*45)
    print("  📊 回测绩效报告")
    print("="*45)
    print(f"  总收益率:    {m['total_return']:>8.2f}%")
    print(f"  CAGR:        {m['cagr']:>8.2f}%")
    print(f"  年化波动率:  {m['vol_ann']:>8.2f}%")
    print(f"  Sharpe:      {m['sharpe']:>8.3f}")
    print(f"  Sortino:     {m['sortino']:>8.3f}")
    print(f"  最大回撤:    {m['max_drawdown']:>8.2f}%")
    print(f"  Calmar:      {m['calmar']:>8.3f}")
    if m["win_rate"] is not None:
        print(f"  胜率:        {m['win_rate']:>8.2f}%")
        print(f"  盈亏比:      {m['profit_factor']:>8.3f}")
        print(f"  交易次数:    {m['num_trades']:>8d}")
    print("="*45 + "\n")
