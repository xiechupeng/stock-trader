"""
实盘/纸交易入口
python live_trade.py --model lstm --interval 3600

环境变量:
  ALPACA_API_KEY    = your_key
  ALPACA_SECRET_KEY = your_secret

默认纸交易 (paper-api.alpaca.markets)
"""
import argparse
import os
from config import CFG
from models.model_factory import load_model
from trading.alpaca_executor import run_live_trading
from trading.signal_generator import SignalGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="lstm",
                        choices=["markov", "lstm", "transformer"])
    parser.add_argument("--symbols",  nargs="+", default=CFG.symbols)
    parser.add_argument("--interval", type=int,  default=3600,
                        help="扫描间隔（秒），默认1小时")
    parser.add_argument("--dry_run",  action="store_true",
                        help="只打印信号，不实际下单")
    args = parser.parse_args()

    # 检查 API Key
    if not CFG.alpaca_key and not args.dry_run:
        print("❌ 未配置 ALPACA_API_KEY，请设置环境变量或使用 --dry_run")
        return

    print(f"\n{'='*50}")
    print(f"  模式: {'DRY RUN (只看信号)' if args.dry_run else '实盘执行 (Paper Trading)'}")
    print(f"  模型: {args.model.upper()}")
    print(f"  股票池: {args.symbols}")
    print(f"  扫描间隔: {args.interval}s")
    print(f"{'='*50}\n")

    print("[1/2] 加载模型...")
    model = load_model(args.model)

    if args.dry_run:
        print("[2/2] DRY RUN — 扫描最新信号...\n")
        from data.fetcher import fetch_ohlcv
        from datetime import datetime, timedelta
        generator = SignalGenerator(model)
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        signals = generator.scan_universe(args.symbols, start=start, end=end)
        print(f"\n共 {len(signals)} 个信号:")
        for s in signals:
            print(f"  [{s.action}] {s.symbol:6s} | token={s.predicted_token} | conf={s.confidence:.2%}")
    else:
        print("[2/2] 启动实盘交易循环...")
        run_live_trading(model, args.symbols, args.interval)


if __name__ == "__main__":
    main()
