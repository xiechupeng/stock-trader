"""
训练入口
python train.py --model [markov|lstm|transformer]
python train.py --model lstm --intraday          ← 使用5分钟数据（27.5万条）
"""
import argparse
import numpy as np
from config import CFG, INTRADAY_TOKEN_CONFIG
from tokenizer.combined import CombinedTokenizer, make_sequences
from models.model_factory import create_model


def build_dataset_daily(symbols: list[str], start: str, end: str):
    """日线数据：yfinance（无限速，10年+历史）"""
    from data.fetcher import fetch_ohlcv
    tokenizer = CombinedTokenizer()
    dfs = {}
    for sym in symbols:
        try:
            dfs[sym] = fetch_ohlcv(sym, start, end, cache_dir=CFG.cache_dir)
        except Exception as e:
            print(f"  [warn] {sym}: {e}")
    return _tokenize_and_split(dfs, tokenizer, label="days")


def build_dataset_intraday(symbols: list[str], start: str = None, end: str = None):
    """5分钟数据：从 cn_us_trader/data_cache/us_5min/ 读取，用日内专属阈值"""
    from data.intraday_fetcher import load_5min_multi, AVAILABLE_5MIN
    use_syms = [s for s in symbols if s in AVAILABLE_5MIN]
    if not use_syms:
        print(f"  [warn] 以下股票无5min数据: {symbols}，改用全部可用: {AVAILABLE_5MIN}")
        use_syms = AVAILABLE_5MIN
    tokenizer = CombinedTokenizer(cfg=INTRADAY_TOKEN_CONFIG)   # ← 5min 阈值
    dfs = load_5min_multi(use_syms, start=start, end=end)
    return _tokenize_and_split(dfs, tokenizer, label="bars")


def _tokenize_and_split(dfs: dict, tokenizer: CombinedTokenizer, label: str):
    all_idxs = []
    for sym, df in dfs.items():
        df_tok = tokenizer.tokenize(df)
        idxs = df_tok["token_idx"].tolist()
        all_idxs.append(idxs)
        print(f"  {sym}: {len(df_tok):,} {label}, {len(set(df_tok['token']))} unique tokens")

    seq_len = CFG.model.sequence_length
    X_all, y_all = [], []
    for idxs in all_idxs:
        X, y = make_sequences(idxs, seq_len)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        raise RuntimeError("数据为空，无法训练")

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    split = int(len(X) * CFG.model.train_ratio)
    return X[:split], y[:split], X[split:], y[split:], all_idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="lstm",
                        choices=["markov", "lstm", "transformer"])
    parser.add_argument("--symbols", nargs="+", default=CFG.symbols)
    parser.add_argument("--train_start", default=CFG.train_start)
    parser.add_argument("--train_end",   default=CFG.train_end)
    parser.add_argument("--intraday", action="store_true",
                        help="使用5分钟数据训练（cn_us_trader/data_cache/us_5min/）")
    args = parser.parse_args()

    mode = "5分钟日内" if args.intraday else "日线"
    print(f"\n{'='*52}")
    print(f"  训练模型: {args.model.upper()}   数据: {mode}")
    print(f"  股票池:   {args.symbols}")
    if not args.intraday:
        print(f"  时间段:   {args.train_start} → {args.train_end}")
    print(f"{'='*52}\n")

    print("[1/3] 构建数据集...")
    if args.intraday:
        X_tr, y_tr, X_val, y_val, sequences = build_dataset_intraday(
            args.symbols,
            start=args.train_start if args.train_start != CFG.train_start else None,
            end=args.train_end   if args.train_end   != CFG.train_end   else None,
        )
    else:
        X_tr, y_tr, X_val, y_val, sequences = build_dataset_daily(
            args.symbols, args.train_start, args.train_end
        )

    print(f"\n  ✅ train={len(X_tr):,}  val={len(X_val):,}  "
          f"vocab tokens seen={len(set(y_tr.tolist())):,}")

    print(f"\n[2/3] 训练 {args.model} ...")
    model = create_model(args.model)

    if args.model == "markov":
        model.fit(sequences)
        acc = model.accuracy(sequences)
        ppl = model.perplexity(sequences)
        print(f"  accuracy={acc:.3f}  perplexity={ppl:.1f}")
        model.save()

    elif args.model in ("lstm", "transformer"):
        mc = CFG.model
        model.fit(
            X_tr, y_tr, X_val, y_val,
            epochs=mc.epochs,
            batch_size=mc.batch_size,
            patience=mc.patience,
        )
        val_acc = model.accuracy(X_val, y_val)
        print(f"\n  ✅ 最终 val_accuracy = {val_acc:.3f}")

    print("\n[3/3] 保存词表...")
    CombinedTokenizer().save_vocab()
    print("  done! → saved_models/vocab.json")
    print(f"\n🎉 训练完成！模型已保存至 saved_models/\n")


if __name__ == "__main__":
    main()
