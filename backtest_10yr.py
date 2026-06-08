"""
10年 Walk-Forward 回测
训练：测试期之前所有数据
测试：每2年滚动向前

python backtest_10yr.py
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xiechupeng/stock_trader')

from data.fetcher import fetch_ohlcv
from swing.zigzag import find_pivots, pivots_to_swings
from swing import swing_tokenizer
from sklearn.ensemble import GradientBoostingClassifier
from backtest.metrics import calc_metrics

SYMS    = ['AMD', 'NVDA', 'GOOGL', 'AAPL', 'AMZN', 'SPY', 'QQQ', 'TSLA', 'META']
ZIGZAG  = 0.03
MAG_THR = 0.09
SEQ_LEN = 12
CONF    = 0.55


def get_all_data():
    cache = {}
    print("加载数据...")
    for sym in SYMS:
        try:
            df = fetch_ohlcv(sym, '2012-01-01', '2026-06-07')
            pivots = find_pivots(df, threshold=ZIGZAG)
            swings = pivots_to_swings(df, pivots)
            vol_avg = float(df['volume'].mean())
            tokens  = [swing_tokenizer.swing_to_token(
                           s, vol_avg,
                           mag_thresh=swing_tokenizer.DAILY_MAG_THRESHOLDS,
                           dur_thresh=swing_tokenizer.DAILY_DUR_THRESHOLDS)
                       for s in swings]
            idxs = [swing_tokenizer.encode(t) for t in tokens]
            cache[sym] = (df, pivots, swings, idxs)
            print(f"  {sym}: {len(df)} days, {len(swings)} swings, "
                  f"avg_mag={np.mean([s['magnitude']*100 for s in swings]):.1f}%")
        except Exception as e:
            print(f"  {sym}: {e}")
    return cache


def build_train(cache, cutoff_date):
    X_all, y_all = [], []
    for sym, (df, pivots, swings, idxs) in cache.items():
        tr_pivots = [(i, p) for i, p in enumerate(pivots)
                     if p.date < pd.Timestamp(cutoff_date) and p.ptype == 'TROUGH']
        for pi_orig, pivot in tr_pivots:
            if pi_orig < SEQ_LEN or pi_orig >= len(swings):
                continue
            label = 1.0 if swings[pi_orig]['magnitude'] >= MAG_THR else 0.0
            X_all.append(idxs[pi_orig - SEQ_LEN : pi_orig])
            y_all.append(label)
    return np.array(X_all, dtype=np.float32), np.array(y_all)


def backtest_period(cache, clf, start, end):
    equities, trades = [], []
    for sym, (df, pivots, swings, idxs) in cache.items():
        n      = len(df)
        closes = df['close'].values
        opens  = df['open'].values
        dates  = df.index
        ma200  = df['close'].rolling(200, min_periods=100).mean().values

        te_pivots = [(i, p) for i, p in enumerate(pivots)
                     if pd.Timestamp(start) <= p.date <= pd.Timestamp(end)]
        if len(te_pivots) < SEQ_LEN + 2:
            continue

        capital, position, eq_curve, sym_trades = 100_000.0, None, [], []

        for pi_orig, pivot in te_pivots:
            if pi_orig < SEQ_LEN or pivot.confirm_idx < 0:
                continue

            act_bar = min(pivot.confirm_idx + 1, n - 1)
            act_p   = float(opens[act_bar])
            act_d   = dates[act_bar]
            cur_c   = float(closes[min(pivot.confirm_idx, n-1)])

            pv = capital + (position['shares'] * cur_c if position else 0)
            eq_curve.append({'date': pivot.confirm_date, 'equity': pv})

            # 止损
            if position and cur_c <= position['stop']:
                net = position['shares'] * cur_c * 0.999 * 0.999
                capital += net
                pnl = (net - position['cost']) / position['cost']
                sym_trades.append({'pnl_pct': pnl, 'reason': 'stop',
                               'entry_price': position['entry_price'],
                               'exit_price': cur_c})
                position = None
                continue

            ctx = np.array([idxs[pi_orig - SEQ_LEN : pi_orig]], dtype=np.float32)

            if position is None and pivot.ptype == 'TROUGH':
                bar_ma = ma200[min(act_bar, n-1)]
                if not (np.isnan(bar_ma) or act_p >= bar_ma * 0.95):
                    continue
                prob = float(clf.predict_proba(ctx)[0, 1])
                if prob >= CONF:
                    slip   = act_p * 1.0005
                    alloc  = capital * 0.20
                    cost   = alloc * 0.999
                    shares = cost / slip
                    capital -= alloc
                    position = {'shares': shares, 'cost': cost,
                                'entry_date': act_d, 'entry_price': slip,
                                'stop': slip * 0.90, 'prob': prob}

            elif position and pivot.ptype == 'PEAK':
                slip = act_p * 0.9995
                net  = position['shares'] * slip * 0.999
                pnl  = (net - position['cost']) / position['cost']
                capital += net
                sym_trades.append({'pnl_pct': pnl, 'reason': 'peak',
                                   'entry_price': position['entry_price'],
                                   'exit_price': slip})
                position = None

        # 强制平仓
        if position:
            last = float(closes[-1])
            net  = position['shares'] * last * 0.999
            capital += net
            sym_trades.append({'pnl_pct': (net - position['cost']) / position['cost'],
                               'reason': 'end', 'entry_price': position['entry_price'],
                               'exit_price': last})

        if eq_curve:
            eq = pd.DataFrame(eq_curve).set_index('date')['equity']
            equities.append(eq)
            trades.extend(sym_trades)  # trades 只用于 win_rate 计算，不需要 entry/exit_price

    if not equities:
        return None, []
    port = pd.concat(equities, axis=1).mean(axis=1)
    return port, trades


def main():
    cache = get_all_data()

    periods = [
        ('2015-2016', '2015-01-01', '2016-12-31'),
        ('2017-2018', '2017-01-01', '2018-12-31'),
        ('2019-2020', '2019-01-01', '2020-12-31'),
        ('2021-2022', '2021-01-01', '2022-12-31'),
        ('2023-2024', '2023-01-01', '2024-12-31'),
        ('2025-2026', '2025-01-01', '2026-06-07'),
    ]

    header = "期间          策略      SPY B&H   超额     Sharpe   MaxDD    胜率   笔数"
    print()
    print(header)
    print('-' * 72)

    all_results = []
    all_eq = []

    for label, start, end in periods:
        # 训练数据：测试期开始前的所有数据
        X_tr, y_tr = build_train(cache, start)
        if len(X_tr) < 30:
            print(f"{label:<14} 训练数据不足，跳过")
            continue

        pos_w = (1 - y_tr.mean()) / (y_tr.mean() + 1e-8)
        sw    = np.where(y_tr == 1, pos_w, 1.0)
        clf   = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42
        )
        clf.fit(X_tr, y_tr, sample_weight=sw)

        port, trades = backtest_period(cache, clf, start, end)
        if port is None:
            continue

        m    = calc_metrics(port, trades)
        spy  = fetch_ohlcv('SPY', start, end)
        spy_r = (spy['close'].iloc[-1] / spy['close'].iloc[0] - 1) * 100
        wr    = m['win_rate'] or 0
        alpha = m['total_return'] - spy_r
        flag  = "✅" if m['total_return'] > 0 else "❌"

        print(f"{label:<14} {m['total_return']:>+7.1f}%  {spy_r:>+7.1f}%  "
              f"{alpha:>+7.1f}%  {m['sharpe']:>+6.3f}  "
              f"{m['max_drawdown']:>6.1f}%  {wr:>5.0f}%  {m['num_trades']:>4}  {flag}")

        all_results.append({
            'period': label, 'strat': m['total_return'],
            'spy': spy_r, 'alpha': alpha,
            'sharpe': m['sharpe'], 'dd': m['max_drawdown'],
            'trades': m['num_trades']
        })
        all_eq.append(port)

    if all_results:
        print('=' * 72)
        total_s = sum(r['strat'] for r in all_results)
        total_b = sum(r['spy']   for r in all_results)
        pos_yrs = sum(1 for r in all_results if r['strat'] > 0)
        print(f"{'합계(11년)':<14} {total_s:>+7.1f}%  {total_b:>+7.1f}%  "
              f"{total_s-total_b:>+7.1f}%  "
              f"{'(年均: '+str(round(total_s/len(all_results),1))+'%)':>16}  "
              f"{'正收益: '+str(pos_yrs)+'/'+str(len(all_results)):>10}")

        # 买入持有实际数据
        print()
        print("各股票 2015→2026 买入持有参考：")
        for sym in ['AMD', 'NVDA', 'SPY', 'AAPL']:
            try:
                df = fetch_ohlcv(sym, '2015-01-01', '2026-06-07')
                r  = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                print(f"  {sym}: {r:+.0f}%  (年化约 {((1+r/100)**(1/11)-1)*100:.0f}%)")
            except:
                pass


if __name__ == '__main__':
    main()
