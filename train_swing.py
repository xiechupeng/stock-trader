"""
Swing Token 模型训练
python train_swing.py
python train_swing.py --threshold 0.005 --seq_len 12
"""
import argparse
import numpy as np
import torch
from data.intraday_fetcher import load_5min_multi, AVAILABLE_5MIN
from swing.swing_dataset import build_dataset
from swing import swing_tokenizer
from models.lstm_model import LSTMModel
from config import CFG


def direction_accuracy(model: LSTMModel,
                        X_val: np.ndarray, y_val: np.ndarray) -> float:
    """预测方向（UP vs DOWN）的准确率——比 token 精确匹配更有意义"""
    model.net.eval()
    x = torch.LongTensor(X_val).to(model.device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            logits = model.net(x[i:i+512])
            preds.extend(logits.argmax(1).cpu().tolist())

    correct = 0
    for pred_idx, true_idx in zip(preds, y_val):
        pd = swing_tokenizer.token_direction(int(pred_idx))
        td = swing_tokenizer.token_direction(int(true_idx))
        if pd == td and pd != 'UNKNOWN':
            correct += 1
    return correct / len(y_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols',    nargs='+', default=AVAILABLE_5MIN)
    parser.add_argument('--threshold',  type=float, default=0.003,
                        help='ZigZag 最小反转幅度（默认0.3%）')
    parser.add_argument('--seq_len',    type=int,   default=10,
                        help='预测用的历史 swing 数量')
    parser.add_argument('--train_end',  default='2025-09-30')
    args = parser.parse_args()

    print(f'\n{"="*54}')
    print(f'  Swing Token 模型训练')
    print(f'  ZigZag threshold : {args.threshold*100:.2f}%')
    print(f'  序列长度         : {args.seq_len} swings')
    print(f'  Vocab size       : {swing_tokenizer.VOCAB_SIZE}')
    print(f'{"="*54}\n')

    # 1. 加载数据
    print('[1/3] 加载5分钟数据并提取 Swing...')
    dfs = load_5min_multi(args.symbols, end=args.train_end)

    X_tr, y_tr, X_val, y_val, token_map = build_dataset(
        dfs,
        threshold=args.threshold,
        seq_len=args.seq_len,
    )
    print(f'\n  train={len(X_tr):,}  val={len(X_val):,}  '
          f'vocab={swing_tokenizer.VOCAB_SIZE}')

    # 检查方向分布
    up_ratio = sum(swing_tokenizer.is_up(swing_tokenizer.decode(int(i))) for i in y_tr) / len(y_tr)
    print(f'  训练集 UP 比例: {up_ratio*100:.1f}%  '
          f'DOWN 比例: {(1-up_ratio)*100:.1f}%')

    # 2. 训练
    print('\n[2/3] 训练 LSTM...')
    mc = CFG.model
    model = LSTMModel(
        vocab_size=swing_tokenizer.VOCAB_SIZE,
        embed_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
    )
    model.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=mc.epochs,
        batch_size=mc.batch_size,
        patience=mc.patience,
        save_path='saved_models/swing_lstm.pt',
    )

    # 3. 评估
    import torch
    token_acc = model.accuracy(X_val, y_val)

    # 幅度预测准确率（核心指标）
    # 真实 label 是 UP_M/L/XL → 预测是否也是 UP_M/L/XL？
    # 只有 UP_L / UP_XL（>0.8%）才值得入场，过滤掉 UP_M 小波段
    def is_worth(tok): return any(tok.startswith(p) for p in ('UP_L_', 'UP_XL_'))
    model.net.eval()
    preds = []
    x_t = torch.LongTensor(X_val).to(model.device)
    with torch.no_grad():
        for i in range(0, len(x_t), 512):
            preds.extend(model.net(x_t[i:i+512]).argmax(1).cpu().tolist())

    # 方向准确率（会是~100%因为ZigZag交替）
    dir_acc = sum(
        swing_tokenizer.token_direction(p) == swing_tokenizer.token_direction(int(t))
        for p, t in zip(preds, y_val)
    ) / len(y_val)

    # 幅度准确率：在 TROUGH 处，能否正确预测大幅 vs 小幅上涨？
    up_samples = [(p, int(t)) for p, t in zip(preds, y_val)
                  if swing_tokenizer.decode(int(t)).startswith('UP_')]
    if up_samples:
        mag_acc = sum(
            is_worth(swing_tokenizer.decode(p)) == is_worth(swing_tokenizer.decode(t))
            for p, t in up_samples
        ) / len(up_samples)
    else:
        mag_acc = 0.0

    worth_base = sum(is_worth(swing_tokenizer.decode(int(t))) for t in y_val
                     if swing_tokenizer.decode(int(t)).startswith('UP_')) / max(1, len(up_samples))

    print(f'\n{"="*52}')
    print(f'  val token accuracy      : {token_acc:.3f}')
    print(f'  val direction accuracy  : {dir_acc:.3f}  (ZigZag交替，非真实信号)')
    print(f'  val magnitude accuracy  : {mag_acc:.3f}  ← 能否预测大波段 vs 小波段')
    print(f'  random magnitude base   : {worth_base:.3f}  (随机猜的正确率)')
    print(f'{"="*52}')
    print(f'\n🎉 模型已保存 → saved_models/swing_lstm.pt\n')


if __name__ == '__main__':
    main()
