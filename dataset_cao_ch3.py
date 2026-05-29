from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


TaskName = Literal["binary", "multiclass5", "multiclass6"]

CLASS6_NAMES = [
    "Preamble_LFMU",
    "LFMD",
    "HFMU",
    "PBPD",
    "IMP",
    "NoiseOnly",
]

CLASS5_NAMES = [
    "Preamble",
    "SCI",
    "PBPD",
    "IMP",
    "Noise",
]


def task_num_classes(task: TaskName) -> int:
    return {"binary": 2, "multiclass5": 5, "multiclass6": 6}[task]


def task_class_names(task: TaskName) -> list[str]:
    if task == "binary":
        return ["NotPreamble", "Preamble"]
    if task == "multiclass5":
        return CLASS5_NAMES.copy()
    if task == "multiclass6":
        return CLASS6_NAMES.copy()
    raise ValueError(f"Unknown task: {task}")


def make_labels(df: pd.DataFrame, task: TaskName) -> np.ndarray:
    if task == "binary":
        return df["flag"].astype(np.int64).to_numpy()

    if task == "multiclass6":
        return (df["class_id"].astype(np.int64).to_numpy() - 1).astype(np.int64)

    if task == "multiclass5":
        family_to_label = {
            "Preamble": 0,
            "SCI": 1,
            "PBPD": 2,
            "IMP": 3,
            "Noise": 4,
        }
        labels = df["class_family"].map(family_to_label)
        if labels.isna().any():
            bad = sorted(df.loc[labels.isna(), "class_family"].astype(str).unique())
            raise ValueError(f"Unknown class_family values: {bad}")
        return labels.astype(np.int64).to_numpy()

    raise ValueError(f"Unknown task: {task}")


class CaoCh3Dataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path = r"F:\gen_data_202601\20260405_NN_10",
        task: TaskName = "binary",
        feature_keys: tuple[str, ...] = ("I_obs", "Q_obs"),
        normalize: bool = True,
        return_meta: bool = False,
        positive_only: bool = False,
        num_threads: int = 8,
        max_rows: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.task = task
        self.feature_keys = tuple(feature_keys)
        self.normalize = normalize
        self.return_meta = return_meta

        df = pd.read_csv(self.csv_path)
        if positive_only:
            df = df[df["flag"].astype(int) == 1].copy()
        if max_rows is not None:
            df = df.head(max_rows).copy()
        self.df = df.reset_index(drop=True)

        self.labels = torch.from_numpy(make_labels(self.df, task)).long()
        self.has_signal = torch.from_numpy(self.df["flag"].astype(np.float32).to_numpy()).float()
        self.gt_sample = torch.from_numpy(self.df["gt_sample"].astype(np.int64).to_numpy()).long()
        self.event_sample = torch.from_numpy(self.df["event_sample"].astype(np.int64).to_numpy()).long()
        self.snrs = self.df["SNR"].astype(np.int64).to_numpy() if "SNR" in self.df.columns else None
        self.file_names = self.df["file"].astype(str).tolist()
        self.h_ids = self.df["h_id"].astype(str).tolist() if "h_id" in self.df.columns else None
        self.win_ids = self.df["win_id"].astype(np.int64).to_numpy() if "win_id" in self.df.columns else None
        self.class_names = self.df["class_name"].astype(str).tolist() if "class_name" in self.df.columns else None
        self.class_families = self.df["class_family"].astype(str).tolist() if "class_family" in self.df.columns else None

        self.file_paths = [self.root_dir / f for f in self.file_names]
        for path in self.file_paths[:10]:
            if not path.exists():
                raise FileNotFoundError(path)

        print(f"[CaoCh3Dataset] Loading {len(self.file_paths)} MAT files from {self.root_dir}")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            loaded = list(
                tqdm(
                    pool.map(self._load_one, self.file_paths),
                    total=len(self.file_paths),
                    desc="Loading MAT",
                )
            )

        self.data: dict[str, torch.Tensor] = {}
        for key in self.feature_keys:
            arr = np.stack([item[key] for item in loaded], axis=0)
            self.data[key] = torch.from_numpy(arr).float()

        print("[CaoCh3Dataset] Done.")
        for key, value in self.data.items():
            print(f"  {key}: {tuple(value.shape)}")
        print(f"  labels: {tuple(self.labels.shape)} task={self.task}")

    def _load_one(self, path: Path) -> dict[str, np.ndarray]:
        mat = sio.loadmat(path)
        sample: dict[str, np.ndarray] = {}
        for key in self.feature_keys:
            if key == "corr_iqa":
                x = np.stack(
                    [
                        mat["I_corr"].astype(np.float32).reshape(-1),
                        mat["Q_corr"].astype(np.float32).reshape(-1),
                        mat["A_corr"].astype(np.float32).reshape(-1),
                    ],
                    axis=0,
                )
            else:
                if key not in mat:
                    raise KeyError(f"{os.path.basename(path)} missing variable {key}")
                x = mat[key].astype(np.float32).reshape(-1)[None, :]

            if self.normalize:
                denom = np.max(np.abs(x))
                if denom > 1e-8:
                    x = x / denom
            sample[key] = x.astype(np.float32)
        return sample

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: tensor[idx] for key, tensor in self.data.items()}
        item["label"] = self.labels[idx]
        item["has_signal"] = self.has_signal[idx]
        item["gt_sample"] = self.gt_sample[idx]
        item["event_sample"] = self.event_sample[idx]
        if self.snrs is not None:
            item["snr"] = int(self.snrs[idx])
        if self.return_meta:
            item["file"] = self.file_names[idx]
            if self.h_ids is not None:
                item["h_id"] = self.h_ids[idx]
            if self.win_ids is not None:
                item["win_id"] = int(self.win_ids[idx])
            if self.class_names is not None:
                item["class_name"] = self.class_names[idx]
            if self.class_families is not None:
                item["class_family"] = self.class_families[idx]
        return item


def make_iq_input(batch: dict, device: str) -> torch.Tensor:
    i_obs = batch["I_obs"].to(device).squeeze(1)
    q_obs = batch["Q_obs"].to(device).squeeze(1)
    return torch.stack([i_obs, q_obs], dim=1)
