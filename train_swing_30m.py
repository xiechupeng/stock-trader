"""
30分钟 Swing 二分类训练
  ZigZag threshold: 0.3%（同5min）
  正样本: 下一段 UP swing > mag_threshold（默认1.2%，约50%正样本率）
  模型: GradientBoosting（小数据更稳定）

python train_swing_30m.py
python train_swing_30m.py --mag_threshold 0.012 --seq_len 12
"""
import argparse, os, numpy as np, joblib
from pathlib import Path
from swing.zigzag import find_pivots, pivots_to_swings, zigzag_stats
from swing import swing_tokenizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score

# 30分钟专用阈值
THRESHOLDS_30M = dict(
    mag_thresh=[0.003, 0.008, 0.020],   # 同5min幅度分档
    dur_thresh=[2, 6],                   # F≤2bars(1hr) N≤6bars(3hr) W>6bars
    vol_thresh=[0.7, 1.5],
)
CACHE_DIR_30M = "data_cache/30min"
ZIGZAG_THRESH = 0.003


def load_30m(symbol: str) -> "pd.DataFrame":
    import pandas as pd
    path = os.path.join(CACHE_DIR_30M, f"{symbol}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"请先运行 download_30m.py: {path}")
    return pd.read_parquet(path)


def build_dataset(symbols: list, train_end: str,
                  mag_threshold: float, seq_len: int,
                  train_ratio: float = 0.9):
    import pandas as pd
    X_all, y_all = [], []

    for sym in symbols:
        try:
            df = load_30m(sym)
            df = df[df.index <= train_end]
            if len(df) < 500:
                print(f"  {sym}: 数据不足，跳过")
                continue
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
            continue

        pivots = find_pivots(df, threshold=ZIGZAG_THRESH)
        swings = pivots_to_swings(df, pivots)
        if len(swings) < seq_len + 2:
            continue

        vol_avg = float(df["volume"].mean())
        tokens  = [swing_tokenizer.swing_to_token(s, vol_avg, **THRESHOLDS_30M)
                   for s in swings]
        idxs    = [swing_tokenizer.encode(t) for t in tokens]

        stats  = zigzag_stats(pivots, swings)
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
        print(f"  {sym}: {stats['n_swings']} swings  "
              f"avg_mag={stats['avg_mag_%']:.2f}%  "
              f"pos={n_pos}/{total} ({n_pos/max(1,total)*100:.0f}%)")

    if not X_all:
        raise RuntimeError("无有效数据，请先运行 download_30m.py")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="+",
                        default=None,  # 自动检测已下载的
                        help="不填则使用所有已下载的30min数据")
    parser.add_argument("--train_end",     default="2025-09-30")
    parser.add_argument("--mag_threshold", type=float, default=0.012,
                        help="正样本阈值：UP swing > 此幅度（默认1.2%）")
    parser.add_argument("--seq_len",       type=int,   default=12)
    args = parser.parse_args()

    # 自动检测已下载的股票
    if args.symbols is None:
        available = [f.stem for f in Path(CACHE_DIR_30M).glob("*.parquet")]
        if not available:
            print("❌ 未找到30min数据，请先运行: python download_30m.py")
            return
        args.symbols = sorted(available)

    print(f"\n{'='*54}")
    print(f"  30分钟 Swing 二分类训练")
    print(f"  ZigZag threshold: {ZIGZAG_THRESH*100:.1f}%")
    print(f"  大波段定义: UP swing > {args.mag_threshold*100:.1f}%")
    print(f"  序列长度: {args.seq_len} swings")
    print(f"  股票池: {args.symbols}")
    print(f"{'='*54}\n")

    print("[1/3] 构建数据集...")
    X_tr, y_tr, X_val, y_val = build_dataset(
        args.symbols, args.train_end,
        mag_threshold=args.mag_threshold,
        seq_len=args.seq_len,
    )
    pos_rate = y_tr.mean()
    print(f"\n  train={len(X_tr):,}  val={len(X_val):,}")
    print(f"  正样本率: train={pos_rate*100:.1f}%  val={y_val.mean()*100:.1f}%")

    print("\n[2/3] 训练 GradientBoosting...")
    pos_weight = (1 - pos_rate) / (pos_rate + 1e-8)
    sample_weight = np.where(y_tr == 1, pos_weight, 1.0)

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        min_samples_leaf=15, random_state=42,
    )
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(clf, "saved_models/swing_30m_gbm.pkl")

    print("\n[3/3] 评估 @ 各置信度...")
    probs    = clf.predict_proba(X_val)[:, 1]
    pos_base = y_val.mean()
    print(f"  随机基准 precision: {pos_base*100:.1f}%\n")

    for thresh in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = (probs >= thresh).astype(float)
        pp = preds.sum()
        tp = ((preds == 1) & (y_val == 1)).sum()
        ap = y_val.sum()
        prec = tp / pp if pp > 0 else 0
        rec  = tp / ap if ap > 0 else 0
        lift = prec / pos_base if pos_base > 0 else 0
        print(f"  thresh={thresh:.2f} | n={int(pp):5d} | "
              f"prec={prec:.3f} | rec={rec:.3f} | lift={lift:.2f}x")

    print(f"\n🎉 模型 → saved_models/swing_30m_gbm.pkl\n")


if __name__ == "__main__":
    main()
