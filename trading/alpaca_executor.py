"""
Alpaca 实盘执行层（纸交易 / 实盘均可）
依赖: pip install alpaca-trade-api
"""
import time
import logging
from datetime import datetime
from typing import Optional

from config import TradingConfig, CFG
from trading.signal_generator import Signal

logger = logging.getLogger(__name__)


class AlpacaExecutor:

    def __init__(self, api_key: str = None, secret_key: str = None,
                 base_url: str = None, cfg: TradingConfig = None):
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError("pip install alpaca-trade-api")

        self.api = tradeapi.REST(
            key_id     = api_key    or CFG.alpaca_key,
            secret_key = secret_key or CFG.alpaca_secret,
            base_url   = base_url   or CFG.alpaca_url,
        )
        self.cfg = cfg or CFG.trading

    # ── 账户信息 ──────────────────────────────────────
    def get_account(self) -> dict:
        a = self.api.get_account()
        return {
            "equity":         float(a.equity),
            "cash":           float(a.cash),
            "buying_power":   float(a.buying_power),
            "portfolio_value": float(a.portfolio_value),
        }

    def get_positions(self) -> list[dict]:
        positions = self.api.list_positions()
        return [
            {
                "symbol":   p.symbol,
                "qty":      float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "side": p.side,
            }
            for p in positions
        ]

    # ── 下单 ──────────────────────────────────────────
    def execute_signal(self, signal: Signal) -> Optional[dict]:
        """
        根据 Signal 执行买入或卖出
        返回 order_id 或 None
        """
        if signal.action == "HOLD":
            return None

        acct     = self.get_account()
        cash     = acct["cash"]
        positions = {p["symbol"]: p for p in self.get_positions()}
        n_pos    = len(positions)

        if signal.action == "BUY":
            # 检查持仓上限
            if n_pos >= self.cfg.max_positions:
                logger.info(f"[skip] {signal.symbol} — max_positions reached ({n_pos})")
                return None
            if signal.symbol in positions:
                logger.info(f"[skip] {signal.symbol} — already holding")
                return None

            alloc_cash = cash * self.cfg.position_size_pct
            price      = self._get_last_price(signal.symbol)
            if price <= 0:
                return None
            qty = int(alloc_cash / price)
            if qty <= 0:
                logger.info(f"[skip] {signal.symbol} — insufficient cash")
                return None

            order = self._submit_order(
                symbol=signal.symbol, qty=qty, side="buy",
                stop_loss=price * (1 - self.cfg.stop_loss_pct),
                take_profit=price * (1 + self.cfg.take_profit_pct),
            )
            logger.info(f"[BUY] {signal.symbol} x{qty} @ ~${price:.2f}")
            return order

        elif signal.action == "SELL":
            if signal.symbol not in positions:
                logger.info(f"[skip] {signal.symbol} — no position to sell")
                return None
            qty = abs(positions[signal.symbol]["qty"])
            order = self.api.submit_order(
                symbol=signal.symbol, qty=qty,
                side="sell", type="market",
                time_in_force="day",
            )
            logger.info(f"[SELL] {signal.symbol} x{qty}")
            return {"order_id": order.id}

        return None

    def close_all_positions(self):
        """一键平仓（风控用）"""
        self.api.close_all_positions()
        logger.warning("[risk] All positions closed!")

    # ── 私有方法 ──────────────────────────────────────
    def _get_last_price(self, symbol: str) -> float:
        try:
            bars = self.api.get_latest_bar(symbol)
            return float(bars.c)
        except Exception as e:
            logger.error(f"price error {symbol}: {e}")
            return 0.0

    def _submit_order(self, symbol, qty, side, stop_loss=None, take_profit=None) -> dict:
        """带止盈止损的 bracket order"""
        if stop_loss and take_profit:
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="market", time_in_force="gtc",
                order_class="bracket",
                stop_loss={"stop_price": round(stop_loss, 2)},
                take_profit={"limit_price": round(take_profit, 2)},
            )
        else:
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="market", time_in_force="day",
            )
        return {"order_id": order.id, "status": order.status}


# ─────────────────────────────────────────────────────
# 定时扫描 + 执行（主循环）
# ─────────────────────────────────────────────────────
def run_live_trading(model, symbols: list[str], interval_seconds: int = 3600):
    """
    每隔 interval_seconds 扫描信号并执行
    建议在美股交易时段 (9:30-16:00 ET) 运行
    """
    from trading.signal_generator import SignalGenerator

    generator = SignalGenerator(model)
    executor  = AlpacaExecutor()

    print(f"[live] 开始实盘交易 | 股票池: {symbols}")
    print(f"[live] 账户状态: {executor.get_account()}\n")

    while True:
        now = datetime.now()
        print(f"\n[{now:%Y-%m-%d %H:%M}] 扫描信号...")

        signals = generator.scan_universe(symbols)
        for sig in signals:
            result = executor.execute_signal(sig)
            if result:
                print(f"  ✅ 订单已提交: {result}")

        # 打印当前持仓
        positions = executor.get_positions()
        if positions:
            print(f"\n  当前持仓 ({len(positions)}):")
            for p in positions:
                print(f"    {p['symbol']:6s} | qty={p['qty']:.0f} | "
                      f"PnL={p['unrealized_pnl']:+.2f}")

        print(f"  下次扫描: {interval_seconds//60} 分钟后")
        time.sleep(interval_seconds)
