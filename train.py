"""
训练入口
python train.py --model [markov|lstm|transformer] --symbols AAPL MSFT NVDA
"""
import argparse
import numpy as np
from config import CFG
from data.fetcher import fetch_multi
from tokenizer.combined import CombinedTokenizer, make_sequences
from models.model_factory import create_model


def build_dataset(symbols: list[str], start: str, end: str):
    """下载+tokenize+切滑窗，返回 (X_train, y_train, X_val, y_val, sequences)"""
    tokenizer = CombinedTokenizer()
    dfs = fetch_multi(symbols, start, end, cache_dir=CFG.cache_dir)

    all_idxs = []
    for sym, df in dfs.items():
        df_tok = tokenizer.tokenize(df)
        all_idxs.append(df_tok["token_idx"].tolist())
        print(f"  {sym}: {len(df_tok)} days, {len(set(df_tok['token']))} unique tokens")

    seq_len = CFG.model.sequence_length
    X_all, y_all = [], []
    for idxs in all_idxs:
        X, y = make_sequences(idxs, seq_len)
        X_all.append(X)
        y_all.append(y)

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
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  训练模型: {args.model.upper()}")
    print(f"  股票池:   {args.symbols}")
    print(f"  时间段:   {args.train_start} → {args.train_end}")
    print(f"{'='*50}\n")

    print("[1/3] 构建数据集...")
    X_tr, y_tr, X_val, y_val, sequences = build_dataset(
        args.symbols, args.train_start, args.train_end
    )
    print(f"  train={len(X_tr):,}  val={len(X_val):,}  vocab={len(set(y_tr)):,}")

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
    tokenizer = CombinedTokenizer()
    tokenizer.save_vocab()
    print("  done! → saved_models/vocab.json")

    print(f"\n🎉 训练完成！模型已保存至 saved_models/\n")


if __name__ == "__main__":
    main()
