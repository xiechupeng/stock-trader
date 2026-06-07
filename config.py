"""
全局配置 — 所有超参数集中管理，修改这里即可
"""
import os
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────
# 1. Tokenization 阈值
# ─────────────────────────────────────────────────────
@dataclass
class TokenConfig:
    # 价格动作阈值（日收益率）
    strong_up: float   = 0.015    # > +1.5%  → U2
    weak_up: float     = 0.003    # +0.3%~+1.5% → U1
    weak_down: float   = -0.003   # -0.3%~-1.5% → D1
    strong_down: float = -0.015   # < -1.5%  → D2

    # 成交量相对于滚动均值
    high_vol_ratio: float = 1.5   # > 1.5x → HIGH
    low_vol_ratio: float  = 0.7   # < 0.7x → LOW
    vol_avg_window: int   = 20

    # K线形态阈值
    big_body_ratio: float      = 0.65  # body/range > 65% → BIG
    doji_body_ratio: float     = 0.10  # body/range < 10% → DOJI
    hammer_shadow_ratio: float = 2.0   # lower_shadow / body > 2 → HAMMER
    gap_threshold: float       = 0.005 # 跳空 > 0.5%


# ─────────────────────────────────────────────────────
# 2. 模型超参数
# ─────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    sequence_length: int = 30      # 用过去30个token预测下一个

    # Markov Chain（order=1 防止 trigram 稀疏；训练数据充足后可升到2）
    markov_order: int = 1          # bigram

    # LSTM
    lstm_embed_dim: int   = 64
    lstm_hidden_dim: int  = 256
    lstm_num_layers: int  = 2
    lstm_dropout: float   = 0.25

    # Transformer
    tf_d_model: int       = 128
    tf_nhead: int         = 4
    tf_num_layers: int    = 4
    tf_dim_ff: int        = 512
    tf_dropout: float     = 0.1

    # 训练
    batch_size: int       = 128
    learning_rate: float  = 1e-3
    epochs: int           = 100
    patience: int         = 12    # early stopping
    train_ratio: float    = 0.9


# ─────────────────────────────────────────────────────
# 3. 交易规则
# ─────────────────────────────────────────────────────
@dataclass
class TradingConfig:
    min_confidence: float = 0.10   # Markov bigram: top-1 通常在 0.10~0.20，此为合理下界

    # 触发买入的 token（下一根K线预期为强势上涨）
    buy_tokens: List[str] = field(default_factory=lambda: [
        "U2_BIG_BULL_HIGH",
        "U2_BIG_BULL_NORMAL",
        "U2_BIG_BULL_LOW",
        "U2_ENGULF_BULL_HIGH",
        "U2_ENGULF_BULL_NORMAL",
        "U2_GAP_UP_HIGH",
        "U2_GAP_UP_NORMAL",
        "U1_BIG_BULL_HIGH",
        "U1_HAMMER_HIGH",
        "U1_HAMMER_NORMAL",
        "U1_ENGULF_BULL_HIGH",
    ])

    # 触发卖出/做空的 token
    sell_tokens: List[str] = field(default_factory=lambda: [
        "D2_BIG_BEAR_HIGH",
        "D2_BIG_BEAR_NORMAL",
        "D2_BIG_BEAR_LOW",
        "D2_ENGULF_BEAR_HIGH",
        "D2_ENGULF_BEAR_NORMAL",
        "D2_GAP_DOWN_HIGH",
        "D2_GAP_DOWN_NORMAL",
        "D1_BIG_BEAR_HIGH",
        "D1_SHOOTING_STAR_HIGH",
        "D1_ENGULF_BEAR_HIGH",
    ])

    # 仓位管理
    position_size_pct: float = 0.20  # 每笔 20% 资金
    max_positions: int       = 5     # 最多同时持仓5只

    # 风险控制
    stop_loss_pct: float    = 0.05   # 止损 -5%
    take_profit_pct: float  = 0.12   # 止盈 +12%
    max_hold_days: int      = 15     # 强制平仓天数


# ─────────────────────────────────────────────────────
# 4. 回测参数
# ─────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission: float      = 0.001    # 0.1% 单边
    slippage: float        = 0.0005   # 0.05% 滑点


# ─────────────────────────────────────────────────────
# 5. 总配置
# ─────────────────────────────────────────────────────
@dataclass
class Config:
    token:    TokenConfig    = field(default_factory=TokenConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    trading:  TradingConfig  = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # 股票池
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "TSLA", "JPM", "V", "SPY", "QQQ",
    ])

    # 时间区间
    train_start: str = "2015-01-01"
    train_end:   str = "2022-12-31"
    test_start:  str = "2023-01-01"
    test_end:    str = "2024-12-31"

    # Alpaca API（paper trading 默认）
    alpaca_key:     str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret:  str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_url:     str = "https://paper-api.alpaca.markets"

    # 文件路径
    model_dir:   str = "saved_models"
    cache_dir:   str = "data_cache"
    results_dir: str = "results"


# 全局单例
CFG = Config()
