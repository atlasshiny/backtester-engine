import numpy as np
import pandas as pd

class EventView:
    __slots__ = ("_arrays", "_idx", "_columns")

    def __init__(self, arrays: dict[str, np.ndarray], idx: int, columns):
        self._arrays = arrays
        self._idx = idx
        self._columns = columns

    def __getattr__(self, name: str):
        try:
            return self._arrays[name][self._idx]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._arrays[self._columns[key]][self._idx]
        return self._arrays[key][self._idx]

    def __len__(self):
        return len(self._columns)

class _HistoryILoc:
    __slots__ = ("_h",)

    def __init__(self, history_view: "HistoryView"):
        self._h = history_view

    def __getitem__(self, item):
        if isinstance(item, int):
            n = len(self._h)
            i = item + n if item < 0 else item
            if i < 0 or i >= n:
                raise IndexError("history.iloc index out of range")
            return EventView(self._h._arrays, self._h._start + i, self._h._columns)
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self._h))
            if step != 1:
                raise ValueError("history.iloc slicing only supports step=1")
            return HistoryView(self._h._arrays, self._h._start + start, self._h._start + stop, self._h._columns)
        raise TypeError("Unsupported indexer for history.iloc")

class HistoryView:
    __slots__ = ("_arrays", "_start", "_end", "_columns", "iloc")

    def __init__(self, arrays: dict[str, np.ndarray], start: int, end: int, columns):
        self._arrays = arrays
        self._start = start
        self._end = end
        self._columns = columns
        self.iloc = _HistoryILoc(self)

    def __len__(self):
        return self._end - self._start

    @property
    def columns(self):
        return self._columns

    @property
    def shape(self):
        return (len(self), len(self._columns))

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self._columns[key]
        return self._arrays[key][self._start:self._end]

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({col: self._arrays[col][self._start:self._end] for col in self._columns})
