from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StationStore:
    def __init__(self, data_root):
        self.root = Path(data_root).resolve()
        metadata = pd.read_csv(self.root / "shared/stations.csv", dtype={"sid": str})
        self.stations = []
        for row in metadata.itertuples():
            path = self.root / "shared/features" / f"{row.sid}.npy"
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if features.ndim != 2 or features.shape[1] != 49 or features.dtype != np.float32:
                raise ValueError(f"Unexpected feature shape/dtype: {path}")
            self.stations.append(SimpleNamespace(sid=str(row.sid), features=features,
                                  capacity=float(row.cap), unit_power=float(row.unit)))
        self.by_sid = {station.sid: i for i, station in enumerate(self.stations)}
        if len(self.by_sid) != len(self.stations):
            raise ValueError("Duplicate station IDs")


def scaler(data_root):
    with np.load(Path(data_root) / "shared/scaler.npz", allow_pickle=False) as values:
        mean, std = values["mean"].copy(), values["std"].copy()
    if mean.shape != (49,) or std.shape != (49,) or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("Invalid archived training scaler")
    return dict(mean=mean, std=std)


def windows(data_root, split, prediction_selection=False):
    frame = pd.read_parquet(Path(data_root) / split / "windows.parquet")
    frame["sid"] = frame.sid.astype(str)
    if prediction_selection:
        if split != "val" or "role" not in frame:
            raise ValueError("Prediction-selection role must be read from validation")
        frame = frame.loc[frame.role.eq("pred_select")].reset_index(drop=True)
    if not len(frame) or frame.duplicated(["sid", "win_start"]).any():
        raise ValueError("Empty or duplicate window membership")
    return frame


class Windows(Dataset):
    def __init__(self, store, index, mean, std):
        self.store, self.mean, self.std = store, mean, std
        ids = index.sid.map(store.by_sid)
        if ids.isna().any():
            raise ValueError("Unknown station in window list")
        self.ids = np.column_stack([ids.to_numpy(np.int64), index.win_start.to_numpy(np.int64)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid, start = self.ids[i]
        station = self.store.stations[sid]
        end = start + 72
        if start < 0 or end + 3 > len(station.features):
            raise IndexError("Window outside its station feature array")
        x = torch.from_numpy(((station.features[start:end] - self.mean) / self.std).astype(np.float32))
        y = torch.from_numpy(station.features[end:end+3, -1].astype(np.float32))
        return x, y, station.capacity, station.unit_power, float(station.features[end-1, -1]), sid, start
