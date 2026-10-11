import numpy as np
from pandas import DataFrame, Index, Series


class ZScorePositions:
    """Entry/exit rules on a z-score, with the position of each asset carried
    from one step to the next.

      - flat: enter short (-1) when entryZ < z < stopZ, long (+1) when
        -stopZ < z < -entryZ, only if the stationarity p-value < pValueMax;
      - in a trade: exit when |z| < exitZ (reverted), or stop out when z goes
        past stopZ against the position. The p-value is not checked to exit;
      - after a stop: no new entry until |z| has come back under exitZ;
      - NaN z (no view, e.g. the OU fit is not mean-reverting): keep the state.
    """

    def __init__(self, entryZ: float = 1.5, exitZ: float = 0.5, stopZ: float = 3.0, pValueMax: float = 0.05):
        self.entryZ, self.exitZ, self.stopZ, self.pValueMax = entryZ, exitZ, stopZ, pValueMax
        self.reset()

    def reset(self):
        self._state = {}   # key -> (position, stopped) after _lastDate
        self._before = {}  # key -> (position, stopped) before _lastDate
        self._lastDate = None

    def nextPosition(self, state: tuple[int, bool], z: float, pValue: float) -> tuple[int, bool]:
        """(position, stopped) after seeing today's z-score, from yesterday's."""
        position, stopped = state
        if np.isnan(z):
            return state
        if position != 0:
            if abs(z) < self.exitZ:
                return 0, False
            if -position * z > self.stopZ:
                return 0, True
            return state
        if stopped:
            return 0, abs(z) >= self.exitZ
        if pValue < self.pValueMax and self.entryZ < abs(z) < self.stopZ:
            return -int(np.sign(z)), False
        return 0, False

    def update(self, dates: Index, z: Series, pValue: Series) -> DataFrame:
        """State after the last of `dates`, one row per key of `z`, columns
        `position` and `stopped`. `dates` is the whole history the strategy
        was handed: the state carries over only when its second to last date
        is the last date seen (the next step of the same run), is recomputed
        from the same start when the last date is seen again, and starts flat
        otherwise (a new run)."""
        date = dates[-1]
        if date == self._lastDate:
            start = self._before
        elif len(dates) > 1 and dates[-2] == self._lastDate:
            start = self._state
        else:
            start = {}
        state = {key: self.nextPosition(start.get(key, (0, False)), z[key], pValue[key]) for key in z.index}
        self._before, self._state, self._lastDate = start, state, date
        return DataFrame(state, index=["position", "stopped"]).T.astype(float)
