"""
Swing 二分类训练
  正样本：下一个 UP swing 幅度 > mag_threshold（默认1.5%）
  负样本：下一个 UP swing 幅度 ≤ mag_threshold

python train_swing_binary.py
python train_swing_binary.py --mag_threshold 0.015 --seq_len 15
"""
import argparse
import numpy as np
from data.intraday_fetcher import load_5min_multi, AVAILABLE_5MIN
from swing.zigzag import find_pivots, pivots_to_swings
from swing import swing_tokenizer
from swing.binary_model import BinarySwingModel
from config import CFG


def build_binary_dataset(dfs: dict, zigzag_thresh: float = 0.003,
                          mag_threshold: float = 0.015,
                          seq_len: int = 15,
                          train_ratio: float = 0.9):
    """
    构建二分类数据集：只包含 TROUGH 处的预测样本
    y=1: 下一段 UP swing 幅度 > mag_threshold
    y=0: 下一段 UP swing 幅度 ≤ mag_threshold
    """
    X_all, y_all = [], []

    for sym, df in dfs.items():
        pivots = find_pivots(df, threshold=zigzag_thresh)
        swings = pivots_to_swings(df, pivots)
        vol_avg = float(df['volume'].mean())

        tokens = [swing_tokenizer.swing_to_token(s, vol_avg) for s in swings]
        idxs   = [swing_tokenizer.encode(t) for t in tokens]

        # 只在 TROUGH 处采样（pi 为偶数时 = trough，奇数 = peak）
        n_pos, n_neg = 0, 0
        for pi, pivot in enumerate(pivots[:-1]):   # 最后一个 pivot 没有下一段
            if pivot.ptype != 'TROUGH':
                continue
            if pi < seq_len:
                continue
            # 下一段是 swings[pi]（UP swing）
            if pi >= len(swings):
                continue
            up_mag = swings[pi]['magnitude']   # 下一段幅度
            label  = 1.0 if up_mag >= mag_threshold else 0.0
            ctx    = idxs[pi - seq_len : pi]
            X_all.append(ctx)
            y_all.append(label)
            if label == 1: n_pos += 1
            else:          n_neg += 1

        print(f"  {sym}: {n_pos} pos / {n_neg} neg  "
              f"({n_pos/(n_pos+n_neg)*100:.1f}% large swings)")

    if not X_all:
        raise RuntimeError("无样本")

    X = np.array(X_all, dtype=np.int64)
    y = np.array(y_all, dtype=np.float32)

    # 时序划分（不shuffle）
    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="+", default=AVAILABLE_5MIN)
    parser.add_argument("--zigzag_thresh", type=float, default=0.003)
    parser.add_argument("--mag_threshold", type=float, default=0.015,
                        help="正样本阈值：UP swing > 此幅度视为大波段（默认1.5%）")
    parser.add_argument("--seq_len",       type=int,   default=15)
    parser.add_argument("--train_end",     default="2025-09-30")
    args = parser.parse_args()

    print(f"\n{'='*54}")
    print(f"  Swing 二分类训练")
    print(f"  大波段定义: UP swing > {args.mag_threshold*100:.1f}%")
    print(f"  序列长度  : {args.seq_len} swings")
    print(f"{'='*54}\n")

    print("[1/3] 构建数据集...")
    dfs = load_5min_multi(args.symbols, end=args.train_end)
    X_tr, y_tr, X_val, y_val = build_binary_dataset(
        dfs,
        zigzag_thresh=args.zigzag_thresh,
        mag_threshold=args.mag_threshold,
        seq_len=args.seq_len,
    )

    pos_rate_tr  = y_tr.mean()
    pos_rate_val = y_val.mean()
    pos_weight   = (1 - pos_rate_tr) / (pos_rate_tr + 1e-8)  # 自动计算权重

    print(f"\n  train={len(X_tr):,}  val={len(X_val):,}")
    print(f"  正样本率: train={pos_rate_tr*100:.1f}%  val={pos_rate_val*100:.1f}%")
    print(f"  pos_weight: {pos_weight:.2f}")

    print(f"\n[2/3] 训练二分类 LSTM (pos_weight={pos_weight:.1f})...")
    mc = CFG.model
    model = BinarySwingModel(
        vocab_size=swing_tokenizer.VOCAB_SIZE,
        embed_dim=32, hidden_dim=128, num_layers=2, dropout=0.2,
        pos_weight=pos_weight,
    )
    model.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=mc.epochs,
        batch_size=256,
        patience=mc.patience,
        save_path="saved_models/swing_binary.pt",
    )

    # 最终评估
    probs = np.array([model.predict_proba(list(x)) for x in X_val])
    for thresh in [0.30, 0.40, 0.50, 0.60]:
        preds = (probs >= thresh).astype(float)
        tp = ((preds == 1) & (y_val == 1)).sum()
        pp = (preds == 1).sum()
        ap = (y_val == 1).sum()
        prec = tp / pp   if pp > 0 else 0
        rec  = tp / ap   if ap > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        print(f"  thresh={thresh:.2f} | n_signal={pp:4d} | prec={prec:.3f} | rec={rec:.3f} | F1={f1:.3f}")

    print(f"\n🎉 模型 → saved_models/swing_binary.pt\n")


if __name__ == "__main__":
    main()
