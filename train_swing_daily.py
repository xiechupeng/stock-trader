"""
日线 Swing 二分类训练
  数据: Yahoo Finance 10年日线（远多于5分钟数据）
  ZigZag threshold: 3%（日线合适，过滤噪音）
  正样本: 下一段 UP swing > mag_threshold（默认5%）

python train_swing_daily.py
python train_swing_daily.py --mag_threshold 0.05 --seq_len 12
"""
import argparse
import numpy as np
from data.fetcher import fetch_ohlcv
from swing.zigzag import find_pivots, pivots_to_swings, zigzag_stats
from swing import swing_tokenizer
from swing.binary_model import BinarySwingModel
from config import CFG

SYMBOLS_DAILY = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "JPM", "V", "SPY", "QQQ",
    "AMD", "NFLX", "HOOD", "SCHW",
]

DAILY_ZIGZAG_THRESH = 0.03   # 3%


def build_daily_dataset(symbols: list, train_start: str, train_end: str,
                         zigzag_thresh: float = 0.03,
                         mag_threshold: float = 0.05,
                         seq_len: int = 12,
                         train_ratio: float = 0.9,
                         cache_dir: str = "data_cache"):
    """
    日线 ZigZag 二分类数据集
    y=1: 下一段 UP swing > mag_threshold
    y=0: 下一段 UP swing ≤ mag_threshold
    """
    X_all, y_all = [], []

    for sym in symbols:
        try:
            df = fetch_ohlcv(sym, train_start, train_end, cache_dir=cache_dir)
        except Exception as e:
            print(f"  [warn] {sym}: {e}")
            continue

        pivots = find_pivots(df, threshold=zigzag_thresh)
        swings = pivots_to_swings(df, pivots)
        if len(swings) < seq_len + 2:
            print(f"  {sym}: swing 不足 ({len(swings)})，跳过")
            continue

        vol_avg = float(df["volume"].mean())
        tokens  = [swing_tokenizer.swing_to_token(
                       s, vol_avg,
                       mag_thresh=swing_tokenizer.DAILY_MAG_THRESHOLDS,
                       dur_thresh=swing_tokenizer.DAILY_DUR_THRESHOLDS)
                   for s in swings]
        idxs    = [swing_tokenizer.encode(t) for t in tokens]

        stats = zigzag_stats(pivots, swings)
        n_pos = n_neg = 0

        for pi, pivot in enumerate(pivots[:-1]):
            if pivot.ptype != "TROUGH" or pi < seq_len:
                continue
            if pi >= len(swings):
                continue
            up_mag = swings[pi]["magnitude"]
            label  = 1.0 if up_mag >= mag_threshold else 0.0
            X_all.append(idxs[pi - seq_len : pi])
            y_all.append(label)
            if label == 1: n_pos += 1
            else:          n_neg += 1

        total = n_pos + n_neg
        print(f"  {sym}: {stats['n_swings']} swings  "
              f"avg_mag={stats['avg_mag_%']:.1f}%  "
              f"pos={n_pos}/{total} ({n_pos/total*100:.0f}% large)")

    if not X_all:
        raise RuntimeError("无有效样本，请检查日期范围或数据")

    X = np.array(X_all, dtype=np.int64)
    y = np.array(y_all, dtype=np.float32)
    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="+", default=SYMBOLS_DAILY)
    parser.add_argument("--train_start",   default="2015-01-01")
    parser.add_argument("--train_end",     default="2024-12-31")
    parser.add_argument("--zigzag_thresh", type=float, default=DAILY_ZIGZAG_THRESH)
    parser.add_argument("--mag_threshold", type=float, default=0.05,
                        help="正样本: UP swing > 此幅度（默认5%）")
    parser.add_argument("--seq_len",       type=int,   default=12)
    args = parser.parse_args()

    print(f"\n{'='*56}")
    print(f"  日线 Swing 二分类训练")
    print(f"  ZigZag threshold : {args.zigzag_thresh*100:.0f}%")
    print(f"  大波段定义       : UP swing > {args.mag_threshold*100:.0f}%")
    print(f"  序列长度         : {args.seq_len} swings")
    print(f"  训练区间         : {args.train_start} → {args.train_end}")
    print(f"{'='*56}\n")

    print("[1/3] 构建数据集...")
    X_tr, y_tr, X_val, y_val = build_daily_dataset(
        args.symbols, args.train_start, args.train_end,
        zigzag_thresh=args.zigzag_thresh,
        mag_threshold=args.mag_threshold,
        seq_len=args.seq_len,
    )

    pos_rate = y_tr.mean()
    pos_weight = (1 - pos_rate) / (pos_rate + 1e-8)
    print(f"\n  train={len(X_tr):,}  val={len(X_val):,}")
    print(f"  正样本率: train={pos_rate*100:.1f}%  val={y_val.mean()*100:.1f}%")
    print(f"  pos_weight: {pos_weight:.2f}")

    print(f"\n[2/3] 训练 LSTM (pos_weight={pos_weight:.1f})...")
    mc = CFG.model
    model = BinarySwingModel(
        vocab_size=swing_tokenizer.VOCAB_SIZE,
        embed_dim=32, hidden_dim=256, num_layers=2, dropout=0.25,
        pos_weight=pos_weight,
    )
    model.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=mc.epochs,
        batch_size=256,
        patience=mc.patience,
        save_path="saved_models/swing_daily_binary.pt",
    )

    print("\n[3/3] 评估 precision/recall @ 各置信度...")
    probs = np.array([model.predict_proba(list(x)) for x in X_val])
    pos_base = y_val.mean()
    print(f"  随机基准 precision: {pos_base*100:.1f}%\n")
    for thresh in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = (probs >= thresh).astype(float)
        pp = preds.sum()
        tp = ((preds == 1) & (y_val == 1)).sum()
        ap = y_val.sum()
        prec = tp / pp   if pp > 0 else 0
        rec  = tp / ap   if ap > 0 else 0
        lift = prec / pos_base if pos_base > 0 else 0
        print(f"  thresh={thresh:.2f} | n={int(pp):4d} | "
              f"prec={prec:.3f} | rec={rec:.3f} | lift={lift:.2f}x")

    print(f"\n🎉 模型 → saved_models/swing_daily_binary.pt\n")


if __name__ == "__main__":
    main()
