from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_cao_ch3 import CaoCh3Dataset, TaskName, make_iq_input, task_class_names, task_num_classes
from models_cao_ch3 import build_model, model_feature_keys


@dataclass
class TrainConfig:
    task: TaskName = "binary"
    model_name: str = "CVWavLeNet1D"
    data_root: str = r"F:\gen_data_202601\20260405_NN_10"
    train_csv: str = str(Path(__file__).resolve().parent / "outputs" / "splits" / "train.csv")
    val_csv: str = str(Path(__file__).resolve().parent / "outputs" / "splits" / "val.csv")
    output_dir: str = str(Path(__file__).resolve().parent / "outputs" / "runs")
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 20260430
    early_stop_patience: int = 12
    early_stop_min_delta: float = 1e-4
    max_rows: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_model(model: torch.nn.Module, batch: dict, model_name: str, device: str) -> torch.Tensor:
    if model_name in {"CVWavLeNet1D", "CV-NET", "CVM-LeNet", "RealIQCNN1D"}:
        x = make_iq_input(batch, device)
        return model(x)
    elif model_name == "MT-DCTNet-IQ":
        iq = make_iq_input(batch, device)
        return model(iq=iq)
    elif model_name == "MT-DCTNet-Corr":
        x_corr = batch["A_corr"].to(device)
        return model(x_corr=x_corr)
    elif model_name in {"MT-DCTNet-Dual", "MT-DCTNet"}:
        x_corr = batch["A_corr"].to(device)
        iq = make_iq_input(batch, device)
        return model(x_corr=x_corr, iq=iq)
    elif model_name == "CorrCNN1D":
        x = batch["A_corr"].to(device)
        return model(x)
    elif model_name in {"CC-Net", "CC-MTL-Net", "CC-CV-MTL-SNet"}:
        x_corr = batch["A_corr"].to(device)
        return model(x_corr)
    elif model_name in {"CC-CV-MTL-Net", "CC-CV-MTL-GNet"}:
        x_corr = batch["A_corr"].to(device)
        iq = make_iq_input(batch, device)
        return model(x_corr, iq)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    average = "binary" if len(class_names) == 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    for idx, name in enumerate(class_names):
        out[f"{name}_precision"] = float(per_p[idx])
        out[f"{name}_recall"] = float(per_r[idx])
        out[f"{name}_f1"] = float(per_f1[idx])
        out[f"{name}_support"] = int(support[idx])
    return out


def build_loader(csv_path: str, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    ds = CaoCh3Dataset(
        csv_path=csv_path,
        root_dir=cfg.data_root,
        task=cfg.task,
        feature_keys=model_feature_keys(cfg.model_name),
        normalize=True,
        return_meta=False,
        num_threads=8,
        max_rows=cfg.max_rows,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0

    desc = "Train" if is_train else "Val"
    for batch in tqdm(loader, desc=desc, leave=False):
        labels = batch["label"].to(cfg.device)
        with torch.set_grad_enabled(is_train):
            logits = forward_model(model, batch, cfg.model_name, cfg.device)
            loss = F.cross_entropy(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pred = torch.argmax(logits, dim=1)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = compute_metrics(y_true, y_pred, task_class_names(cfg.task))
    return total_loss / max(total_n, 1), metrics


def save_history(history: list[dict], run_dir: Path) -> None:
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_accuracy"], label="val_accuracy")
    plt.plot(df["epoch"], df["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "metrics.png", dpi=200)
    plt.close()


def run_training(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    num_classes = task_num_classes(cfg.task)
    run_name = f"{cfg.task}_{cfg.model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_loader = build_loader(cfg.train_csv, cfg, shuffle=True)
    val_loader = build_loader(cfg.val_csv, cfg, shuffle=False)

    model = build_model(cfg.model_name, num_classes=num_classes).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = -math.inf
    stale = 0
    history: list[dict] = []
    best_path = run_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")
        train_loss, train_metrics = run_epoch(model, train_loader, cfg, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, cfg, None)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)
        save_history(history, run_dir)
        print(
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_acc={val_metrics['accuracy']:.5f} val_f1={val_metrics['f1']:.5f}"
        )

        if val_metrics["f1"] - best_f1 > cfg.early_stop_min_delta:
            best_f1 = val_metrics["f1"]
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "model_name": cfg.model_name,
                    "task": cfg.task,
                    "num_classes": num_classes,
                    "class_names": task_class_names(cfg.task),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            (run_dir / "best_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
            print(f"Saved best checkpoint: {best_path}")
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print(f"Early stop after {stale} stale epochs.")
                break

    result = {"run_dir": str(run_dir), "best_path": str(best_path), "best_f1": best_f1}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multiclass5", "multiclass6"], default="binary")
    parser.add_argument(
        "--model-name",
        choices=[
            "CVWavLeNet1D",
            "CV-NET",
            "CVM-LeNet",
            "RealIQCNN1D",
            "CorrCNN1D",
            "CC-Net",
            "CC-MTL-Net",
            "CC-CV-MTL-Net",
            "CC-CV-MTL-SNet",
            "CC-CV-MTL-GNet",
            "MT-DCTNet-IQ",
            "MT-DCTNet-Corr",
            "MT-DCTNet-Dual",
            "MT-DCTNet",
        ],
        default="CVWavLeNet1D",
    )
    parser.add_argument("--data-root", default=TrainConfig.data_root)
    parser.add_argument("--train-csv", default=TrainConfig.train_csv)
    parser.add_argument("--val-csv", default=TrainConfig.val_csv)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default=TrainConfig.device)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    result = run_training(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
