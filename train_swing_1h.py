"""
1小时 Swing 二分类训练
数据来源：30min parquet → pandas resample("1h") → 1小时 OHLCV

python train_swing_1h.py
"""
import argparse, os, numpy as np, joblib, pandas as pd
from pathlib import Path
from swing.zigzag import find_pivots, pivots_to_swings, zigzag_stats
from swing import swing_tokenizer
from sklearn.ensemble import GradientBoostingClassifier
from train_swing_30m import CACHE_DIR_30M

ZIGZAG_THRESH = 0.003   # 0.3%，同30min

# 1小时专用 token 阈值
THRESHOLDS_1H = dict(
    mag_thresh=[0.003, 0.008, 0.020],  # 同5min/30min幅度分档
    dur_thresh=[2, 5],                  # F≤2bars(2hr) N≤5bars(5hr) W>5bars
    vol_thresh=[0.7, 1.5],
)


def load_1h(symbol: str) -> pd.DataFrame:
    """读取 30min parquet，聚合成 1小时"""
    path = os.path.join(CACHE_DIR_30M, f"{symbol}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"请先运行 download_30m.py: {path}")
    df30 = pd.read_parquet(path)

    # 重采样：每2根30min bar → 1根1h bar
    df1h = df30[["open","high","low","close","volume"]].resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    # 重新计算衍生列
    df1h["returns"]      = df1h["close"].pct_change()
    df1h["log_returns"]  = np.log(df1h["close"] / df1h["close"].shift(1))
    df1h["vol_avg"]      = df1h["volume"].rolling(13).mean()  # 13根1h≈2交易日
    df1h["vol_ratio"]    = df1h["volume"] / df1h["vol_avg"]
    df1h["range"]        = df1h["high"] - df1h["low"]
    df1h["body"]         = (df1h["close"] - df1h["open"]).abs()
    df1h["body_ratio"]   = df1h["body"] / df1h["range"].replace(0, np.nan)
    df1h["upper_shadow"] = df1h.apply(lambda r: r["high"]-max(r["open"],r["close"]), axis=1)
    df1h["lower_shadow"] = df1h.apply(lambda r: min(r["open"],r["close"])-r["low"], axis=1)
    df1h.dropna(subset=["returns"], inplace=True)
    df1h.index.name = "date"
    return df1h


def build_dataset(symbols, train_end, mag_threshold, seq_len, train_ratio=0.9):
    X_all, y_all = [], []
    for sym in symbols:
        try:
            df = load_1h(sym)
            df = df[df.index <= train_end]
            if len(df) < 200:
                continue
        except FileNotFoundError as e:
            print(f"  [warn] {e}"); continue

        pivots = find_pivots(df, threshold=ZIGZAG_THRESH)
        swings = pivots_to_swings(df, pivots)
        if len(swings) < seq_len + 2:
            continue

        vol_avg = float(df["volume"].mean())
        tokens  = [swing_tokenizer.swing_to_token(s, vol_avg, **THRESHOLDS_1H)
                   for s in swings]
        idxs    = [swing_tokenizer.encode(t) for t in tokens]
        stats   = zigzag_stats(pivots, swings)

        n_pos = n_neg = 0
        for pi, pivot in enumerate(pivots[:-1]):
            if pivot.ptype != "TROUGH" or pi < seq_len or pi >= len(swings):
                continue
            label = 1.0 if swings[pi]["magnitude"] >= mag_threshold else 0.0
            X_all.append(idxs[pi - seq_len : pi])
            y_all.append(label)
            if label == 1: n_pos += 1
            else:          n_neg += 1

        total = n_pos + n_neg
        print(f"  {sym:<6}: {stats['n_swings']:4d} swings  "
              f"avg={stats['avg_mag_%']:.2f}%  "
              f"pos={n_pos}/{total} ({n_pos/max(1,total)*100:.0f}%)")

    if not X_all:
        raise RuntimeError("无数据，请先运行 download_30m.py")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="+", default=None)
    parser.add_argument("--train_end",     default="2025-09-30")
    parser.add_argument("--mag_threshold", type=float, default=0.012,
                        help="正样本阈值：UP swing > 此幅度（默认1.2%）")
    parser.add_argument("--seq_len",       type=int, default=12)
    args = parser.parse_args()

    if args.symbols is None:
        args.symbols = sorted(f.stem for f in Path(CACHE_DIR_30M).glob("*.parquet"))

    print(f"\n{'='*54}")
    print(f"  1小时 Swing 二分类训练（from 30min resample）")
    print(f"  ZigZag threshold : {ZIGZAG_THRESH*100:.1f}%")
    print(f"  大波段定义       : UP swing > {args.mag_threshold*100:.1f}%")
    print(f"  序列长度         : {args.seq_len} swings")
    print(f"  股票池           : {args.symbols}")
    print(f"{'='*54}\n")

    # 先显示 1h swing 统计
    print("[0] 1小时 Swing 统计...")
    for sym in args.symbols[:5]:
        try:
            df = load_1h(sym)
            pivots = find_pivots(df, threshold=ZIGZAG_THRESH)
            swings = pivots_to_swings(df, pivots)
            mags = [s["magnitude"]*100 for s in swings if s["direction"]=="UP"]
            if mags:
                print(f"  {sym:<6}: {len(swings):4d} swings  "
                      f"avg={np.mean(mags):.2f}%  "
                      f"median={np.median(mags):.2f}%  "
                      f"p75={np.percentile(mags,75):.2f}%")
        except: pass
    print()

    print("[1/3] 构建数据集...")
    X_tr, y_tr, X_val, y_val = build_dataset(
        args.symbols, args.train_end, args.mag_threshold, args.seq_len
    )
    pos_rate = y_tr.mean()
    pos_weight = (1 - pos_rate) / (pos_rate + 1e-8)
    print(f"\n  train={len(X_tr):,}  val={len(X_val):,}")
    print(f"  正样本率: train={pos_rate*100:.1f}%  val={y_val.mean()*100:.1f}%")
    print(f"  pos_weight: {pos_weight:.2f}")

    print("\n[2/3] 训练 GradientBoosting...")
    sample_weight = np.where(y_tr == 1, pos_weight, 1.0)
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        min_samples_leaf=15, random_state=42,
    )
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(clf, "saved_models/swing_1h_gbm.pkl")

    print("\n[3/3] 评估 @ 各置信度...")
    probs    = clf.predict_proba(X_val)[:, 1]
    pos_base = y_val.mean()
    print(f"  随机基准 precision: {pos_base*100:.1f}%\n")

    best_thresh, best_lift = 0, 0
    for thresh in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        preds = (probs >= thresh).astype(float)
        pp = preds.sum()
        tp = ((preds == 1) & (y_val == 1)).sum()
        ap = y_val.sum()
        prec = tp / pp if pp > 0 else 0
        rec  = tp / ap if ap > 0 else 0
        lift = prec / pos_base if pos_base > 0 else 0
        if lift > best_lift and pp >= 20:
            best_lift, best_thresh = lift, thresh
        print(f"  thresh={thresh:.2f} | n={int(pp):5d} | "
              f"prec={prec:.3f} | rec={rec:.3f} | lift={lift:.2f}x")

    print(f"\n  推荐置信度阈值: {best_thresh:.2f}  (lift={best_lift:.2f}x)")
    print(f"\n🎉 模型 → saved_models/swing_1h_gbm.pkl\n")


if __name__ == "__main__":
    main()
