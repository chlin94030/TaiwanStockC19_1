"""Reproducible V19 research configuration. No order execution."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json

VERSION = '19.0.0'
TZ = timezone(timedelta(hours=8))
TECH = ('trend', 'relative_strength', 'breakout', 'reversal', 'pattern', 'volume')
OPTIONAL = ('revenue', 'quality', 'foreign', 'trust', 'joint', 'dealer', 'margin', 'holders')
FACTORS = TECH + OPTIONAL
# Starting hypotheses, NOT optimized coefficients. Missing optional groups are
# excluded from a common training universe rather than being invented as zero.
PRIORS = {
    5: dict(trend=12, relative_strength=12, breakout=16, reversal=14, pattern=10, volume=12,
            revenue=7, quality=3, foreign=3, trust=5, joint=3, dealer=1, margin=1, holders=1),
    20: dict(trend=16, relative_strength=14, breakout=10, reversal=10, pattern=8, volume=8,
             revenue=12, quality=8, foreign=4, trust=4, joint=3, dealer=1, margin=1, holders=1),
    60: dict(trend=18, relative_strength=12, breakout=7, reversal=5, pattern=6, volume=5,
             revenue=16, quality=16, foreign=4, trust=4, joint=3, dealer=1, margin=1, holders=2),
}

@dataclass(frozen=True)
class Settings:
    years: int = 3
    stocks: int = 100
    dates: int = 10
    horizons: tuple[int, ...] = (5, 20, 60)
    trials: int = 24
    seed: int = 1900
    top_k: int = 5
    warmup: int = 80
    train_dates: int = 35
    min_universe: int = 10
    min_turnover: float = 5_000_000.0
    max_entry_gap: float = 0.04
    buy_fee: float = 0.001425
    sell_fee: float = 0.001425
    sell_tax: float = 0.003
    benchmark_sell_tax: float = 0.001
    slippage: float = 0.0005
    optional_data: bool = False
    request_budget: int = 260
    timeout_seconds: int = 12
    benchmark: str = '0050.TW'
    as_of: str = ''

    def __post_init__(self):
        if not 1 <= self.years <= 3: raise ValueError('years must be 1..3')
        if self.stocks < self.top_k or self.top_k < 1: raise ValueError('invalid stock count')
        if not 2 <= self.dates <= 100: raise ValueError('dates must be 2..100')
        if not self.horizons or any(h < 1 or h > 250 for h in self.horizons): raise ValueError('invalid horizons')
        if self.trials < 2 or self.trials > 100: raise ValueError('trials must be 2..100')
        if self.min_universe < self.top_k: raise ValueError('min_universe below top_k')
        if any(x < 0 or x >= .2 for x in (self.buy_fee, self.sell_fee, self.sell_tax, self.benchmark_sell_tax, self.slippage)):
            raise ValueError('invalid transaction cost')
        if self.as_of: datetime.strptime(self.as_of, '%Y-%m-%d')

    def end(self):
        if self.as_of: return self.as_of
        # Never assume an unfinished same-day candle is a completed daily bar.
        return (datetime.now(TZ).date() - timedelta(days=1)).isoformat()

    def start(self):
        return (datetime.strptime(self.end(), '%Y-%m-%d') - timedelta(days=365*self.years)).date().isoformat()

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def prior(horizon, factors=FACTORS):
    source = PRIORS[min(PRIORS, key=lambda h: abs(h-horizon))]
    selected = {k: float(source.get(k, 0)) for k in factors}
    total = sum(selected.values())
    if total <= 0: raise ValueError('no usable factors')
    return {k: v / total for k, v in selected.items()}
