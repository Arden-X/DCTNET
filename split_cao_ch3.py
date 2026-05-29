from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    index_path: Path = Path(r"F:\gen_data_202601\20260405_NN_10\index.xlsx")
    output_dir: Path = Path(__file__).resolve().parent / "outputs" / "splits"
    train_hids_per_case: int = 15
    val_hids_per_case: int = 5
    test_hids_per_case: int = 5
    seed: int = 20260430


def load_index(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = [
        "win_id",
        "class_id",
        "class_name",
        "class_family",
        "flag",
        "h_id",
        "SNR",
        "gt_sample",
        "file",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["case"] = df["h_id"].astype(str).str.split("-", n=1).str[0].astype(int)
    return df.reset_index(drop=True)


def split_by_hid(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    split_by_hid: dict[str, str] = {}
    expected = cfg.train_hids_per_case + cfg.val_hids_per_case + cfg.test_hids_per_case

    for case_id, sub in df.groupby("case"):
        hids = np.array(sorted(sub["h_id"].astype(str).unique()), dtype=object)
        if len(hids) < expected:
            raise ValueError(f"case={case_id} has {len(hids)} h_id values, expected at least {expected}")
        rng.shuffle(hids)

        train_hids = hids[: cfg.train_hids_per_case]
        val_hids = hids[cfg.train_hids_per_case : cfg.train_hids_per_case + cfg.val_hids_per_case]
        test_hids = hids[
            cfg.train_hids_per_case
            + cfg.val_hids_per_case : cfg.train_hids_per_case
            + cfg.val_hids_per_case
            + cfg.test_hids_per_case
        ]

        for hid in train_hids:
            split_by_hid[str(hid)] = "train"
        for hid in val_hids:
            split_by_hid[str(hid)] = "val"
        for hid in test_hids:
            split_by_hid[str(hid)] = "test"

    out = df.copy()
    out["split"] = out["h_id"].astype(str).map(split_by_hid)
    if out["split"].isna().any():
        raise RuntimeError("Some rows did not receive a split.")
    return out


def summarize(df: pd.DataFrame) -> None:
    print("\nRows by split:")
    print(df["split"].value_counts().sort_index().to_string())

    print("\nUnique h_id by case/split:")
    print(
        df[["case", "h_id", "split"]]
        .drop_duplicates()
        .groupby(["case", "split"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nRows by class/split:")
    print(df.groupby(["class_name", "split"]).size().unstack(fill_value=0).to_string())

    print("\nRows by SNR/split:")
    print(df.groupby(["SNR", "split"]).size().unstack(fill_value=0).to_string())


def write_splits(df: pd.DataFrame, cfg: SplitConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        path = cfg.output_dir / f"{split}.csv"
        df[df["split"] == split].to_csv(path, index=False)
        print(f"Wrote {path}")

    summary_path = cfg.output_dir / "split_summary.csv"
    summary = (
        df.groupby(["split", "class_name", "SNR"])
        .size()
        .reset_index(name="rows")
        .sort_values(["split", "class_name", "SNR"])
    )
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


def parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, default=SplitConfig.index_path)
    parser.add_argument("--output-dir", type=Path, default=SplitConfig.output_dir)
    parser.add_argument("--train-hids-per-case", type=int, default=SplitConfig.train_hids_per_case)
    parser.add_argument("--val-hids-per-case", type=int, default=SplitConfig.val_hids_per_case)
    parser.add_argument("--test-hids-per-case", type=int, default=SplitConfig.test_hids_per_case)
    parser.add_argument("--seed", type=int, default=SplitConfig.seed)
    args = parser.parse_args()
    return SplitConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    df = load_index(cfg.index_path)
    split_df = split_by_hid(df, cfg)
    summarize(split_df)
    write_splits(split_df, cfg)


if __name__ == "__main__":
    main()
