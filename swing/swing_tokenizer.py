"""
Swing Token 词表

格式: {DIR}_{MAG}_{DUR}_{VOL}
  DIR : UP | DN
  MAG : S(小) M(中) L(大) XL(超大)
  DUR : F(快) N(普) W(慢/宽)
  VOL : H(高量) N(普量) L(低量)

2×4×3×3 = 72 有效 token + PAD + UNK = 74
"""

PAD = '<PAD>'
UNK = '<UNK>'

# ── 5分钟 默认阈值 ────────────────────────────────────
MAG_THRESHOLDS = [0.003, 0.008, 0.020]   # S<0.3% M<0.8% L<2% XL≥2%
DUR_THRESHOLDS = [6, 18]                  # F≤6bar(30min) N≤18bar(1.5h) W>18bar
VOL_THRESHOLDS = [0.7, 1.5]              # L<0.7x N<1.5x H≥1.5x

# ── 日线 专用阈值（swing 平均幅度 ~6%，ZigZag thresh=3%）────
DAILY_MAG_THRESHOLDS = [0.03, 0.07, 0.15] # S<3% M<7% L<15% XL≥15%
DAILY_DUR_THRESHOLDS = [3, 10]             # F≤3天 N≤10天 W>10天
DAILY_VOL_THRESHOLDS = [0.7, 1.5]         # 同 5min


def _build_vocab():
    tokens = [PAD, UNK]
    for d in ['UP', 'DN']:
        for m in ['S', 'M', 'L', 'XL']:
            for dur in ['F', 'N', 'W']:
                for v in ['H', 'N', 'L']:
                    tokens.append(f'{d}_{m}_{dur}_{v}')
    t2i = {t: i for i, t in enumerate(tokens)}
    i2t = {i: t for t, i in t2i.items()}
    return t2i, i2t


TOKEN2IDX, IDX2TOKEN = _build_vocab()
VOCAB_SIZE = len(TOKEN2IDX)   # 74


# ── 单 swing → token ─────────────────────────────────
def swing_to_token(swing: dict, session_vol_avg: float,
                   mag_thresh=None, dur_thresh=None, vol_thresh=None) -> str:
    """
    swing dict（来自 zigzag.pivots_to_swings）→ token string
    session_vol_avg: 当日或近期 rolling 均量（用于判断高/低量）
    """
    direction = swing['direction']
    magnitude = swing['magnitude']   # 小数，如 0.005 = 0.5%
    n_bars    = swing['n_bars']
    avg_vol   = swing['avg_vol']

    mt = mag_thresh or MAG_THRESHOLDS
    dt = dur_thresh or DUR_THRESHOLDS
    vt = vol_thresh or VOL_THRESHOLDS

    d = 'UP' if direction == 'UP' else 'DN'

    if magnitude < mt[0]:   m = 'S'
    elif magnitude < mt[1]: m = 'M'
    elif magnitude < mt[2]: m = 'L'
    else:                   m = 'XL'

    if n_bars <= dt[0]:    dur = 'F'
    elif n_bars <= dt[1]:  dur = 'N'
    else:                  dur = 'W'

    ratio = avg_vol / session_vol_avg if session_vol_avg > 1e-8 else 1.0
    if ratio >= vt[1]:    v = 'H'
    elif ratio >= vt[0]:  v = 'N'
    else:                 v = 'L'

    tok = f'{d}_{m}_{dur}_{v}'
    return tok if tok in TOKEN2IDX else UNK


def encode(token: str) -> int:
    return TOKEN2IDX.get(token, TOKEN2IDX[UNK])

def decode(idx: int) -> str:
    return IDX2TOKEN.get(idx, UNK)

def is_up(token: str) -> bool:
    return isinstance(token, str) and token.startswith('UP_')

def is_down(token: str) -> bool:
    return isinstance(token, str) and token.startswith('DN_')

def token_direction(idx: int) -> str:
    """返回 'UP' | 'DOWN' | 'UNKNOWN'"""
    t = decode(idx)
    if is_up(t):   return 'UP'
    if is_down(t): return 'DOWN'
    return 'UNKNOWN'
