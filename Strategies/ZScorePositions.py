import numpy as np
from pandas import DataFrame, Index, Series

FLAT = (0, False, np.nan, np.nan)


class ZScorePositions:
    """Entry/exit rules on a z-score, with the position of each asset carried
    from one step to the next.

    The state of an asset is (position, stopped, entryMu, entryStd).
      - flat: z is measured against today's fit (mu, std). Enter short (-1)
        when entryZ < z < stopZ, long (+1) when -stopZ < z < -entryZ, only if
        the stationarity p-value < pValueMax and the fitted half-life <
        maxHalfLife (a spread that takes months to revert is not worth
        holding). The fit's mu and std are frozen as entryMu and entryStd;
      - in a trade: z is measured against the frozen entryMu and entryStd,
        not today's fit, so a rolling mean drifting toward the price does not
        fake a reversion or dodge the stop, and z stays defined when today's
        fit is not mean-reverting. Exit when |z| < exitZ (reverted), stop out
        when z goes past stopZ against the position. The p-value is not
        checked to exit, nor the half-life;
      - after a stop: no new entry until |z| has come back under exitZ;
      - flat with no valid fit (NaN mu or std): no entry.

    pValueMax defaults to 0.01, not 0.05: the test runs on every asset every
    day, so at 5% about one asset in twenty passes by chance each day.
    """

    def __init__(self, entryZ: float = 1.5, exitZ: float = 0.5, stopZ: float = 3.0,
                 pValueMax: float = 0.01, maxHalfLife: float = np.inf):
        self.entryZ, self.exitZ, self.stopZ, self.pValueMax = entryZ, exitZ, stopZ, pValueMax
        self.maxHalfLife = maxHalfLife
        self.reset()

    def reset(self):
        self._state = {}   # key -> state after _lastDate
        self._before = {}  # key -> state before _lastDate
        self._lastDate = None

    def nextPosition(self, state: tuple, value: float, mu: float, std: float, pValue: float,
                     halfLife: float = 0.0) -> tuple[tuple, float]:
        """State after seeing today's value and fit, from yesterday's state,
        and the z-score the decision was made on."""
        position, stopped, entryMu, entryStd = state
        if position != 0:
            z = (value - entryMu) / entryStd
            if abs(z) < self.exitZ:
                return FLAT, z
            if -position * z > self.stopZ:
                return (0, True, np.nan, np.nan), z
            return state, z

        z = (value - mu) / std
        if np.isnan(z):
            return state, z
        if stopped:
            return (0, abs(z) >= self.exitZ, np.nan, np.nan), z
        if pValue < self.pValueMax and halfLife < self.maxHalfLife and self.entryZ < abs(z) < self.stopZ:
            return (-int(np.sign(z)), False, mu, std), z
        return FLAT, z

    def update(self, dates: Index, value: Series, mu: Series, std: Series, pValue: Series = None,
               halfLife: Series = None) -> DataFrame:
        """State after the last of `dates`, one row per key of `value`, with
        columns `position`, `stopped`, `zScore` (the z the decision used),
        `entryMu` and `entryStd`. `dates` is the whole history the strategy
        was handed: the state carries over only when its second to last date
        is the last date seen (the next step of the same run), is recomputed
        from the same start when the last date is seen again, and starts flat
        otherwise (a new run). Without `pValue` or `halfLife`, that check does
        not gate entries."""
        if pValue is None:
            pValue = Series(0.0, index=value.index)
        if halfLife is None:
            halfLife = Series(0.0, index=value.index)
        date = dates[-1]
        if date == self._lastDate:
            start = self._before
        elif len(dates) > 1 and dates[-2] == self._lastDate:
            start = self._state
        else:
            start = {}
        state, rows = {}, {}
        for key in value.index:
            state[key], z = self.nextPosition(start.get(key, FLAT), value[key], mu[key], std[key],
                                              pValue[key], halfLife[key])
            position, stopped, entryMu, entryStd = state[key]
            rows[key] = {"position": position, "stopped": stopped, "zScore": z,
                         "entryMu": entryMu, "entryStd": entryStd}
        self._before, self._state, self._lastDate = start, state, date
        return DataFrame(rows, index=["position", "stopped", "zScore", "entryMu", "entryStd"]).T.astype(float)
