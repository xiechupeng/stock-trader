"""
Swing Token 回测
逻辑：
  确认 TROUGH → 模型预测下一段是 UP → 买入
  确认 PEAK   → 模型预测下一段是 DOWN → 卖出（或平多仓）

python backtest_swing.py
python backtest_swing.py --threshold 0.003 --confidence 0.45
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.intraday_fetcher import load_5min_multi, AVAILABLE_5MIN
from swing.zigzag import find_pivots, pivots_to_swings
from swing import swing_tokenizer
from swing.swing_dataset import df_to_swing_tokens
from models.lstm_model import LSTMModel
from backtest.metrics import calc_metrics, print_metrics


def backtest_single(df: pd.DataFrame, model: LSTMModel, symbol: str,
                    threshold: float = 0.003,
                    seq_len: int = 10,
                    min_confidence: float = 0.45,
                    commission: float = 0.001,
                    slippage: float = 0.0005,
                    position_size_pct: float = 0.20) -> dict:
    """单只股票 Swing Token 回测"""
    pivots = find_pivots(df, threshold=threshold)
    if len(pivots) < seq_len + 2:
        print(f'  {symbol}: pivot 不足 ({len(pivots)})，跳过')
        return {}

    swings          = pivots_to_swings(df, pivots)
    session_vol_avg = float(df['volume'].mean())

    # 全部 swing → token → idx
    tokens = [swing_tokenizer.swing_to_token(s, session_vol_avg) for s in swings]
    idxs   = [swing_tokenizer.encode(t) for t in tokens]

    closes = df['close'].values
    dates  = df.index

    capital      = 100_000.0
    position     = None
    equity_curve = []
    trades       = []

    for pi, pivot in enumerate(pivots):
        # 跳过未确认的 pivot（序列末尾，confirm_idx=-1）
        if pivot.confirm_idx < 0:
            continue

        # 实盘入场时间：确认 bar 的【下一根】bar，消除 look-ahead bias
        act_bar   = min(pivot.confirm_idx + 1, len(closes) - 1)
        act_price = float(closes[act_bar])
        act_date  = dates[act_bar]

        # 当前净值（以确认时刻估值）
        pv = capital
        if position:
            pv += position['shares'] * float(closes[pivot.confirm_idx])
        equity_curve.append({'date': pivot.confirm_date, 'equity': pv})

        if pi < seq_len:
            continue

        context  = idxs[pi - seq_len : pi]
        pred_idx, conf = model.predict(context)
        pred_tok = swing_tokenizer.decode(pred_idx)

        # 只在预测大波段（UP_L > 0.8% 或 UP_XL > 2%）时买入
        pred_mag_ok = any(pred_tok.startswith(p) for p in ('UP_L_', 'UP_XL_'))

        # ── TROUGH 确认 → 下一根 bar 买入
        if position is None and pivot.ptype == 'TROUGH':
            if pred_mag_ok and conf >= min_confidence:
                slip   = act_price * (1 + slippage)
                alloc  = capital * position_size_pct
                comm   = alloc * commission
                shares = (alloc - comm) / slip
                capital -= alloc
                position = {
                    'entry_price': slip,
                    'entry_date':  act_date,
                    'shares':      shares,
                    'cost':        alloc - comm,
                    'entry_bar':   act_bar,
                }

        # ── PEAK 确认 → 下一根 bar 平仓
        elif position is not None and pivot.ptype == 'PEAK':
            slip     = act_price * (1 - slippage)
            proceeds = position['shares'] * slip
            comm     = proceeds * commission
            net      = proceeds - comm
            pnl_pct  = (net - position['cost']) / position['cost']
            capital += net
            trades.append({
                'symbol':      symbol,
                'entry_price': position['entry_price'],
                'exit_price':  slip,
                'entry_date':  position['entry_date'],
                'exit_date':   act_date,
                'pnl_pct':     pnl_pct,
                'hold_bars':   act_bar - position['entry_bar'],
            })
            position = None

    # 强制平仓
    if position and len(closes) > 0:
        last  = float(closes[-1])
        net   = position['shares'] * last * (1 - slippage) * (1 - commission)
        pnl   = (net - position['cost']) / position['cost']
        capital += net
        trades.append({
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'exit_price':  last,
            'entry_date':  position['entry_date'],
            'exit_date':   dates[-1],
            'pnl_pct':     pnl,
            'hold_bars':   len(closes) - 1 - position['entry_bar'],
        })

    if not equity_curve:
        return {}

    eq = pd.DataFrame(equity_curve).set_index('date')['equity']
    m  = calc_metrics(eq, trades)
    return {'equity': eq, 'trades': trades, 'metrics': m, 'symbol': symbol,
            'n_pivots': len(pivots), 'n_swings': len(swings)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols',    nargs='+', default=AVAILABLE_5MIN)
    parser.add_argument('--threshold',  type=float, default=0.003)
    parser.add_argument('--seq_len',    type=int,   default=10)
    parser.add_argument('--confidence', type=float, default=0.45)
    parser.add_argument('--start',      default='2025-10-01')
    parser.add_argument('--end',        default='2026-06-01')
    args = parser.parse_args()

    print(f'\n{"="*54}')
    print(f'  Swing Token 回测（波峰/波谷策略）')
    print(f'  ZigZag threshold : {args.threshold*100:.2f}%')
    print(f'  最低置信度       : {args.confidence}')
    print(f'  时间段           : {args.start} → {args.end}')
    print(f'{"="*54}\n')

    # 加载模型
    print('[1/3] 加载 Swing LSTM...')
    model = LSTMModel(
        vocab_size=swing_tokenizer.VOCAB_SIZE,
        embed_dim=32, hidden_dim=128, num_layers=2, dropout=0.2,
    )
    model.load('saved_models/swing_lstm.pt')

    # 加载数据
    print('[2/3] 加载5分钟数据...')
    dfs = load_5min_multi(args.symbols, start=args.start, end=args.end)

    # 回测
    print('[3/3] 运行回测...\n')
    results = {}
    for sym, df in dfs.items():
        print(f'── {sym} ──')
        r = backtest_single(df, model, sym,
                            threshold=args.threshold,
                            seq_len=args.seq_len,
                            min_confidence=args.confidence)
        if r:
            print_metrics(r['metrics'])
            results[sym] = r

    # Portfolio 汇总
    if not results:
        print('无有效回测结果')
        return

    equities   = [v['equity'] for v in results.values()]
    all_trades = [t for v in results.values() for t in v['trades']]
    port_eq    = pd.concat(equities, axis=1).mean(axis=1)
    port_m     = calc_metrics(port_eq, all_trades)

    print('\n' + '='*54)
    print('  [Portfolio Summary]')
    print('='*54)
    print_metrics(port_m)

    # 图表
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for sym, r in results.items():
        axes[0].plot(r['equity'], label=sym, alpha=0.7, linewidth=1)
    axes[0].plot(port_eq, color='black', linewidth=2.5, label='Portfolio')
    axes[0].set_title(f'Swing Token 净值曲线  threshold={args.threshold*100:.1f}%  conf={args.confidence}')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    dd = (port_eq - port_eq.cummax()) / port_eq.cummax() * 100
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4, color='red')
    axes[1].set_title('Portfolio 回撤 (%)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'results/swing_backtest.png'
    plt.savefig(path, dpi=150)
    print(f'\n[plot] → {path}')


if __name__ == '__main__':
    main()
