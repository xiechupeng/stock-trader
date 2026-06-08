"""
截面动量 + 趋势过滤策略
================================
逻辑（每月第一个交易日执行）：

1. 趋势过滤：
   - SPY > 200MA → 市场处于上升趋势，正常选股
   - SPY < 200MA → 切换到全仓 TLT（长期国债），避开熊市

2. 选股（市场上升时）：
   - 计算 universe 中每只股票的「风险调整动量」
     score = (price_now / price_126d_ago - 1) / rolling_std_21d
     （6个月涨幅 / 波动率 ≈ 夏普比率）
   - 选 Top N 只（默认10只）等权持有
   - 下月再比较，换掉跌出 Top 的

python momentum_strategy.py --backtest
python momentum_strategy.py --live_signal   # 打印今日信号
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.fetcher import fetch_ohlcv
from backtest.metrics import calc_metrics, print_metrics

# ── 股票池：大中盘美股 + 行业 ETF ──────────────────────
UNIVERSE = [
    # 大盘科技
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
    # 其他成长
    "NFLX","UBER","SHOP","COIN","PLTR","SQ","CRWD","PANW","SNOW",
    # 金融
    "JPM","GS","V","MA","HOOD",
    # 医疗/消费
    "UNH","LLY","COST","WMT",
    # ETF（行业轮动）
    "QQQ","XLK","XLF","XLE","XLV","XLI",
    # 债券（熊市避险）
    "TLT","IEF",
]

SPY_TICKER = "SPY"
BOND_TICKER = "TLT"    # 熊市时持仓

# ── 参数 ──────────────────────────────────────────────
TOP_N        = 10      # 每月持仓只数
MOMENTUM_DAYS = 126    # 动量回看窗口（6个月≈126交易日）
TREND_MA     = 200     # SPY 趋势均线
VOL_WINDOW   = 21      # 波动率窗口（1个月）
REBAL_FREQ   = "ME"    # 月末再平衡（pandas freq）
COMMISSION   = 0.001   # 单边 0.1%
STOP_LOSS    = 0.15    # 单股止损 15%


def fetch_all(symbols: list, start: str, end: str,
              cache_dir: str = "data_cache") -> dict[str, pd.Series]:
    """批量拉收盘价，返回 {sym: close_series}"""
    closes = {}
    for sym in symbols:
        try:
            df = fetch_ohlcv(sym, start, end, cache_dir=cache_dir)
            closes[sym] = df["close"].rename(sym)
        except Exception as e:
            pass  # 数据缺失时跳过
    return closes


def momentum_score(close: pd.Series, lookback: int = 126,
                   vol_window: int = 21) -> float:
    """
    风险调整动量分数（当前日期）
    = 过去 lookback 天涨幅 / 过去 vol_window 天波动率
    """
    if len(close) < lookback + vol_window:
        return np.nan
    ret_126  = close.iloc[-1] / close.iloc[-lookback] - 1
    daily_ret = close.pct_change().iloc[-vol_window:]
    vol       = daily_ret.std() * np.sqrt(252) + 1e-8
    return ret_126 / vol


def run_backtest(start: str = "2015-01-01",
                 end:   str = "2026-06-07",
                 top_n: int = TOP_N,
                 verbose: bool = True) -> dict:

    # ── 拉数据 ───────────────────────────────────────
    all_syms  = list(set(UNIVERSE + [SPY_TICKER, BOND_TICKER]))
    if verbose:
        print(f"[1/4] 拉取 {len(all_syms)} 只股票数据 ({start} → {end})...")
    closes_raw = fetch_all(all_syms, start, end)
    if SPY_TICKER not in closes_raw:
        raise RuntimeError("无法获取 SPY 数据")

    # 对齐日期
    price_df = pd.DataFrame(closes_raw).dropna(how="all")
    spy_close = price_df[SPY_TICKER]

    # ── 确定每月再平衡日期 ────────────────────────────
    rebal_dates = price_df.resample(REBAL_FREQ).last().index
    rebal_dates = [d for d in rebal_dates if d >= price_df.index[MOMENTUM_DAYS + VOL_WINDOW]]

    if verbose:
        print(f"[2/4] 再平衡日期: {len(rebal_dates)} 个月...")

    # ── 回测循环 ─────────────────────────────────────
    capital   = 100_000.0
    holdings  = {}   # {sym: shares}
    cost_basis = {}  # {sym: entry_price}
    equity_curve = []
    trades       = []
    monthly_returns = []

    for i, rebal_date in enumerate(rebal_dates):
        # 当前价格
        try:
            today_prices = price_df.loc[rebal_date]
        except KeyError:
            continue

        # ─ 估值 ─────────────────────────────────────
        port_value = capital
        for sym, shares in holdings.items():
            if sym in today_prices and not np.isnan(today_prices[sym]):
                port_value += shares * today_prices[sym]
        equity_curve.append({"date": rebal_date, "equity": port_value})

        # ─ 趋势过滤：SPY vs 200MA ─────────────────────
        spy_hist = spy_close.loc[:rebal_date].iloc[-TREND_MA:]
        spy_ma200 = spy_hist.mean()
        spy_now   = today_prices[SPY_TICKER]
        in_uptrend = spy_now >= spy_ma200

        # ─ 计算各股动量分数 ──────────────────────────
        scores = {}
        hist   = price_df.loc[:rebal_date]
        for sym in price_df.columns:
            if sym in [SPY_TICKER, BOND_TICKER]:
                continue
            s = hist[sym].dropna()
            sc = momentum_score(s, MOMENTUM_DAYS, VOL_WINDOW)
            if not np.isnan(sc) and sc > 0:   # 只选正动量
                scores[sym] = sc

        # ─ 决定目标持仓 ──────────────────────────────
        if not in_uptrend:
            # 熊市：全切 TLT
            target = {BOND_TICKER: 1.0}
            regime = "BEAR→TLT"
        elif scores:
            top_syms = sorted(scores, key=scores.get, reverse=True)[:top_n]
            target = {sym: 1.0 / top_n for sym in top_syms}
            regime = f"BULL top{top_n}"
        else:
            target = {BOND_TICKER: 1.0}
            regime = "NO_SIGNAL→TLT"

        # ─ 止损检查（当前持仓 vs 成本）─────────────────
        stop_sold = []
        for sym, shares in list(holdings.items()):
            if sym not in today_prices:
                continue
            cur_p   = today_prices[sym]
            cost_p  = cost_basis.get(sym, cur_p)
            drawdown = (cur_p - cost_p) / cost_p
            if drawdown <= -STOP_LOSS:
                # 止损卖出
                proceeds = shares * cur_p * (1 - COMMISSION)
                capital  += proceeds
                trades.append({"sym": sym, "pnl_pct": drawdown, "reason": "stop"})
                stop_sold.append(sym)
        for sym in stop_sold:
            del holdings[sym]
            if sym in cost_basis:
                del cost_basis[sym]

        # ─ 卖出不在目标的股票 ────────────────────────
        to_sell = [s for s in holdings if s not in target]
        for sym in to_sell:
            if sym not in today_prices:
                continue
            cur_p = today_prices[sym]
            proceeds = holdings[sym] * cur_p * (1 - COMMISSION)
            entry_p  = cost_basis.get(sym, cur_p)
            pnl      = (cur_p - entry_p) / entry_p
            capital  += proceeds
            trades.append({"sym": sym, "pnl_pct": pnl,
                           "entry_price": entry_p, "exit_price": cur_p})
            del holdings[sym]
            del cost_basis[sym]

        # ─ 买入新目标 ────────────────────────────────
        # 计算当前持仓市值
        curr_value = capital
        for sym, shares in holdings.items():
            if sym in today_prices:
                curr_value += shares * today_prices[sym]

        for sym, weight in target.items():
            if sym not in today_prices or np.isnan(today_prices[sym]):
                continue
            price    = today_prices[sym]
            alloc    = curr_value * weight
            cur_mktval = holdings.get(sym, 0) * price

            # 只有差异 > 5% 才换仓（减少交易）
            if abs(alloc - cur_mktval) / (curr_value + 1e-8) < 0.05:
                continue

            if alloc > cur_mktval:
                buy_val = alloc - cur_mktval
                if capital < buy_val:
                    buy_val = capital
                shares_buy = buy_val * (1 - COMMISSION) / price
                holdings[sym]   = holdings.get(sym, 0) + shares_buy
                cost_basis[sym] = price
                capital -= buy_val
            else:
                sell_val   = cur_mktval - alloc
                shares_sell = sell_val / price
                proceeds    = shares_sell * price * (1 - COMMISSION)
                entry_p     = cost_basis.get(sym, price)
                pnl         = (price - entry_p) / entry_p
                holdings[sym] -= shares_sell
                if holdings[sym] <= 0:
                    del holdings[sym]
                    if sym in cost_basis:
                        del cost_basis[sym]
                capital += proceeds
                trades.append({"sym": sym, "pnl_pct": pnl,
                               "entry_price": entry_p, "exit_price": price})

    # ── 最终估值 ──────────────────────────────────────
    last_date  = price_df.index[-1]
    last_prices = price_df.iloc[-1]
    final_val   = capital
    for sym, shares in holdings.items():
        if sym in last_prices:
            final_val += shares * last_prices[sym]
    equity_curve.append({"date": last_date, "equity": final_val})

    # ── 指标（月度数据插值到日频后计算，避免Sharpe虚高）──────
    eq_raw = pd.DataFrame(equity_curve).set_index("date")["equity"]
    eq_raw = eq_raw[~eq_raw.index.duplicated()].sort_index()

    # 插值到日频（交易日）
    spy_idx = price_df[SPY_TICKER].index
    eq = eq_raw.reindex(spy_idx).ffill().dropna()

    metrics = calc_metrics(eq, [t for t in trades if "entry_price" in t])

    # ── SPY 买入持有对比 ──────────────────────────────
    spy_bnh = (price_df[SPY_TICKER].iloc[-1] / price_df[SPY_TICKER].iloc[0] - 1) * 100
    qqq_bnh = (price_df["QQQ"].iloc[-1] / price_df["QQQ"].iloc[0] - 1) * 100 \
              if "QQQ" in price_df else None

    return {
        "equity":   eq,
        "metrics":  metrics,
        "trades":   trades,
        "spy_bnh":  spy_bnh,
        "qqq_bnh":  qqq_bnh,
        "price_df": price_df,
    }


def plot_results(result: dict, save_path: str = "results/momentum_strategy.png"):
    eq      = result["equity"]
    metrics = result["metrics"]
    pf      = result["price_df"]

    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 净值曲线
    ax = axes[0]
    eq_norm = eq / eq.iloc[0] * 100
    ax.plot(eq_norm, color="steelblue", lw=2.5, label="Momentum Strategy")
    # SPY
    spy_norm = pf[SPY_TICKER] / pf[SPY_TICKER].iloc[0] * 100
    ax.plot(spy_norm, color="gray", lw=1.5, linestyle="--", label="SPY B&H", alpha=0.8)
    ax.set_title("Momentum Strategy vs SPY (normalized to 100)", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)

    # 回撤
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    axes[1].fill_between(dd.index, dd, 0, alpha=0.5, color="red")
    axes[1].set_title("Drawdown (%)")
    axes[1].grid(True, alpha=0.3)

    # Rolling 12-month return vs SPY
    eq_ret12  = eq.pct_change(252).dropna() * 100
    spy_ret12 = pf[SPY_TICKER].pct_change(252).dropna() * 100
    axes[2].plot(eq_ret12,  color="steelblue", lw=1.5, label="Strategy 12M return")
    axes[2].plot(spy_ret12, color="gray", lw=1, linestyle="--", label="SPY 12M return", alpha=0.7)
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set_title("Rolling 12-month Return (%)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[plot] -> {save_path}")


def print_report(result: dict):
    m = result["metrics"]
    print("\n" + "="*55)
    print("  MOMENTUM STRATEGY 回测报告")
    print("="*55)
    print_metrics(m)
    print(f"  {'─'*45}")
    print(f"  SPY  买入持有:  {result['spy_bnh']:>+.1f}%")
    if result['qqq_bnh']:
        print(f"  QQQ  买入持有:  {result['qqq_bnh']:>+.1f}%")
    strat = m['total_return']
    spy   = result['spy_bnh']
    print(f"  策略 超越 SPY: {strat - spy:>+.1f}%")
    print(f"  {'─'*45}")

    # 年度胜率
    eq = result["equity"]
    annual = eq.resample("YE").last().pct_change().dropna() * 100
    spy_annual = result["price_df"][SPY_TICKER].resample("YE").last().pct_change().dropna() * 100
    print("\n  年度表现:")
    beat_count = 0
    for yr in annual.index:
        s  = annual.get(yr, np.nan)
        sb = spy_annual.get(yr, np.nan)
        if np.isnan(s) or np.isnan(sb):
            continue
        flag = "✅" if s > sb else "❌"
        beat_count += (s > sb)
        print(f"    {yr.year}  策略={s:>+6.1f}%  SPY={sb:>+6.1f}%  {flag}")
    total = len([yr for yr in annual.index if not np.isnan(spy_annual.get(yr, np.nan))])
    print(f"\n  跑赢 SPY: {beat_count}/{total} 年 ({beat_count/total*100:.0f}%)")
    print("="*55 + "\n")


def live_signal():
    """打印今日交易信号（实盘用）"""
    from datetime import datetime, timedelta
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=600)).strftime("%Y-%m-%d")

    closes = fetch_all(list(set(UNIVERSE + [SPY_TICKER])), start, end)
    price_df = pd.DataFrame(closes).dropna(how="all")

    spy_close = price_df[SPY_TICKER]
    spy_ma200 = spy_close.iloc[-TREND_MA:].mean()
    spy_now   = spy_close.iloc[-1]
    in_uptrend = spy_now >= spy_ma200

    print(f"\nSPY {spy_now:.2f}  vs  200MA {spy_ma200:.2f}  "
          f"→ {'上升趋势 ✅' if in_uptrend else '下降趋势 ❌ 持 TLT'}")

    if not in_uptrend:
        print("→ 全仓 TLT（债券）\n")
        return

    scores = {}
    for sym in price_df.columns:
        if sym in [SPY_TICKER, BOND_TICKER]:
            continue
        s  = price_df[sym].dropna()
        sc = momentum_score(s, MOMENTUM_DAYS, VOL_WINDOW)
        if not np.isnan(sc):
            scores[sym] = sc

    top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
    print(f"\n今日 Top {TOP_N} 动量股（各 {100//TOP_N}%）:")
    for i, sym in enumerate(top, 1):
        p = price_df[sym].iloc[-1]
        r6m = (price_df[sym].iloc[-1]/price_df[sym].iloc[-MOMENTUM_DAYS]-1)*100
        print(f"  {i:2d}. {sym:<6}  价格={p:>8.2f}  6M涨幅={r6m:>+7.1f}%  score={scores[sym]:>5.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest",    action="store_true")
    parser.add_argument("--live_signal", action="store_true")
    parser.add_argument("--start",  default="2015-01-01")
    parser.add_argument("--end",    default="2026-06-07")
    parser.add_argument("--top_n",  type=int, default=TOP_N)
    args = parser.parse_args()

    if args.backtest or (not args.live_signal):
        print(f"\n运行回测: {args.start} → {args.end}  Top{args.top_n}\n")
        result = run_backtest(args.start, args.end, args.top_n)
        print_report(result)
        plot_results(result)

    if args.live_signal:
        live_signal()
