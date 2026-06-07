"""
ZigZag 算法 — 识别5分钟日内波峰/波谷

threshold: 最小反转幅度（默认0.3%）
  涨了0.3%才算确认波谷，跌了0.3%才算确认波峰

用 close 价格做检测（简单可靠），返回 Pivot 列表
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Pivot:
    bar_idx:      int            # 极值所在 bar（trough/peak 价格）
    date:         pd.Timestamp   # 极值时间
    price:        float          # 极值价格
    ptype:        str            # 'PEAK' | 'TROUGH'
    confirm_idx:  int = -1       # 确认 bar（价格反转 threshold 时的 bar）
    confirm_date: object = None  # 确认时间
    confirm_price: float = 0.0   # 确认时的 close 价格

    def __repr__(self):
        return (f"Pivot({self.ptype} @ {self.date.strftime('%m-%d %H:%M')} "
                f"price={self.price:.2f} confirmed@bar{self.confirm_idx})")


def find_pivots(df: pd.DataFrame,
                threshold: float = 0.003,
                price_col: str = 'close') -> list[Pivot]:
    """
    在 df 上运行 ZigZag，返回已确认的 Pivot 列表（按时间排序）。

    注意：确认存在滞后 —— 当价格从极值反转 threshold 时才确认前一个 pivot。
    这保证了实盘可用（无未来信息泄露）。
    """
    prices = df[price_col].values
    dates  = df.index
    n = len(prices)
    if n < 3:
        return []

    pivots    = []
    direction = None      # None | 'UP' | 'DOWN'
    ext_idx   = 0
    ext_price = float(prices[0])

    for i in range(1, n):
        p = float(prices[i])

        if direction is None:
            # ── 寻找第一个显著方向
            if p >= ext_price * (1 + threshold):
                pivots.append(Pivot(ext_idx, dates[ext_idx], ext_price, 'TROUGH'))
                ext_idx, ext_price, direction = i, p, 'UP'
            elif p <= ext_price * (1 - threshold):
                pivots.append(Pivot(ext_idx, dates[ext_idx], ext_price, 'PEAK'))
                ext_idx, ext_price, direction = i, p, 'DOWN'
            else:
                # 更新极值（找更极端的起点）
                if p > ext_price:
                    ext_idx, ext_price = i, p
                elif p < ext_price:
                    ext_idx, ext_price = i, p

        elif direction == 'UP':
            if p > ext_price:
                ext_idx, ext_price = i, p      # 继续创新高
            elif p <= ext_price * (1 - threshold):
                # 确认 PEAK：极值在 ext_idx，确认在当前 bar i
                pivots.append(Pivot(
                    ext_idx, dates[ext_idx], ext_price, 'PEAK',
                    confirm_idx=i, confirm_date=dates[i], confirm_price=float(p)
                ))
                ext_idx, ext_price, direction = i, p, 'DOWN'

        elif direction == 'DOWN':
            if p < ext_price:
                ext_idx, ext_price = i, p      # 继续创新低
            elif p >= ext_price * (1 + threshold):
                # 确认 TROUGH：极值在 ext_idx，确认在当前 bar i
                pivots.append(Pivot(
                    ext_idx, dates[ext_idx], ext_price, 'TROUGH',
                    confirm_idx=i, confirm_date=dates[i], confirm_price=float(p)
                ))
                ext_idx, ext_price, direction = i, p, 'UP'

    # 收尾：加最后一个未确认极值（序列完整性用，confirm=-1 标记为未确认）
    if direction == 'UP':
        pivots.append(Pivot(ext_idx, dates[ext_idx], ext_price, 'PEAK'))
    elif direction == 'DOWN':
        pivots.append(Pivot(ext_idx, dates[ext_idx], ext_price, 'TROUGH'))

    return pivots


def pivots_to_swings(df: pd.DataFrame, pivots: list[Pivot]) -> list[dict]:
    """
    将相邻 pivot 对转成 swing 字典列表。
    每个 swing:
      direction  : 'UP'  (trough→peak)  | 'DOWN' (peak→trough)
      magnitude  : abs(end-start)/start  （幅度%）
      n_bars     : 经过的5min bar数
      avg_vol    : 区间平均成交量
      start/end  : Pivot 对象
    """
    swings = []
    for i in range(len(pivots) - 1):
        p1, p2 = pivots[i], pivots[i + 1]
        bars      = df.iloc[p1.bar_idx : p2.bar_idx + 1]
        direction = 'UP' if p2.ptype == 'PEAK' else 'DOWN'
        magnitude = abs(p2.price - p1.price) / p1.price
        n_bars    = max(1, len(bars))
        avg_vol   = float(bars['volume'].mean()) if 'volume' in bars.columns else 1.0
        swings.append({
            'start':     p1,
            'end':       p2,
            'direction': direction,
            'magnitude': magnitude,
            'n_bars':    n_bars,
            'avg_vol':   avg_vol,
        })
    return swings


def zigzag_stats(pivots: list[Pivot], swings: list[dict]) -> dict:
    """打印 ZigZag 统计，方便调参"""
    if not swings:
        return {}
    ups   = [s for s in swings if s['direction'] == 'UP']
    downs = [s for s in swings if s['direction'] == 'DOWN']
    mags  = [s['magnitude'] * 100 for s in swings]
    bars  = [s['n_bars'] for s in swings]
    return {
        'n_pivots':    len(pivots),
        'n_swings':    len(swings),
        'n_up':        len(ups),
        'n_down':      len(downs),
        'avg_mag_%':   round(float(np.mean(mags)), 3),
        'avg_bars':    round(float(np.mean(bars)), 1),
        'med_bars':    round(float(np.median(bars)), 1),
    }
