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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_cao_ch3 import CaoCh3Dataset, make_iq_input, task_class_names, task_num_classes
from models_cao_ch3 import build_model, model_feature_keys


SYNC_MODELS = {"CV-MTL-Net", "CC-MTL-Net", "CC-CV-MTL-Net", "CC-CV-MTL-SNet", "CC-CV-MTL-GNet", "MT-DCTNet"}


@dataclass
class TrainConfig:
    task: str = "multiclass6"
    model_name: str = "CC-CV-MTL-Net"
    data_root: str = r"F:\gen_data_202601\20260405_NN_10"
    train_csv: str = str(Path(__file__).resolve().parent / "outputs" / "splits" / "train.csv")
    val_csv: str = str(Path(__file__).resolve().parent / "outputs" / "splits" / "val.csv")
    output_dir: str = str(Path(__file__).resolve().parent / "outputs" / "runs")
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 20260508
    lambda_cls: float = 1.0
    lambda_loc: float = 3.0
    triangular_half_width: int = 12
    hit_tol: int = 10
    score_loc_weight: float = 0.01
    select_by: str = "joint"
    early_stop_patience: int = 12
    early_stop_min_delta: float = 1e-4
    max_rows: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_model_with_loc(model: torch.nn.Module, batch: dict, model_name: str, device: str):
    if model_name == "CV-MTL-Net":
        iq = make_iq_input(batch, device)
        return model(iq, return_loc=True)
    if model_name in {"CC-MTL-Net", "CC-CV-MTL-SNet"}:
        x_corr = batch["A_corr"].to(device)
        return model(x_corr, return_loc=True)
    if model_name in {"CC-CV-MTL-Net", "CC-CV-MTL-GNet"}:
        x_corr = batch["A_corr"].to(device)
        iq = make_iq_input(batch, device)
        return model(x_corr, iq, return_loc=True)
    if model_name == "MT-DCTNet":
        x_corr = batch["A_corr"].to(device)
        iq = make_iq_input(batch, device)
        return model(x_corr=x_corr, iq=iq, return_loc=True)
    raise ValueError(f"{model_name} does not expose a synchronization head in models_cao_ch3.py")


def build_triangular_targets(gt_sample: torch.Tensor, length: int, half_width: int) -> torch.Tensor:
    positions = torch.arange(length, device=gt_sample.device).unsqueeze(0)
    centers = torch.clamp(gt_sample.float(), min=0, max=length - 1).unsqueeze(1)
    dist = torch.abs(positions - centers)
    targets = 1.0 - dist / float(half_width)
    return torch.clamp(targets, min=0.0, max=1.0)


def loc_metrics(pred: np.ndarray, gt: np.ndarray, hit_tol: int) -> dict:
    if len(gt) == 0:
        return {"loc_mae": math.nan, "loc_rmse": math.nan, f"loc_hit@{hit_tol}": math.nan}
    err = pred.astype(np.float64) - gt.astype(np.float64)
    abs_err = np.abs(err)
    return {
        "loc_mae": float(np.mean(abs_err)),
        "loc_rmse": float(np.sqrt(np.mean(err * err))),
        f"loc_hit@{hit_tol}": float(np.mean(abs_err <= hit_tol)),
    }


def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }
    pp, rr, ff, ss = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    for idx, name in enumerate(class_names):
        out[f"{name}_precision"] = float(pp[idx])
        out[f"{name}_recall"] = float(rr[idx])
        out[f"{name}_f1"] = float(ff[idx])
        out[f"{name}_support"] = int(ss[idx])
    return out


def build_loader(csv_path: str, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    ds = CaoCh3Dataset(
        csv_path=csv_path,
        root_dir=cfg.data_root,
        task=cfg.task,  # type: ignore[arg-type]
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
) -> tuple[dict, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    class_names = task_class_names(cfg.task)  # type: ignore[arg-type]

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_loc_pred: list[np.ndarray] = []
    all_loc_gt: list[np.ndarray] = []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
        labels = batch["label"].to(cfg.device)
        gt_sample = batch["gt_sample"].to(cfg.device)
        pos_mask = labels == 0  # Preamble_LFMU only.

        with torch.set_grad_enabled(is_train):
            cls_logits, loc_logits = forward_model_with_loc(model, batch, cfg.model_name, cfg.device)
            cls_loss = F.cross_entropy(cls_logits, labels)

            if pos_mask.any():
                loc_target = build_triangular_targets(
                    gt_sample=gt_sample[pos_mask],
                    length=loc_logits.size(1),
                    half_width=cfg.triangular_half_width,
                )
                loc_loss = F.binary_cross_entropy_with_logits(loc_logits[pos_mask], loc_target)
            else:
                loc_loss = torch.zeros((), device=cfg.device)

            loss = cfg.lambda_cls * cls_loss + cfg.lambda_loc * loc_loss
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pred = torch.argmax(cls_logits, dim=1)
        loc_pred = torch.argmax(loc_logits, dim=1)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_cls_loss += float(cls_loss.item()) * bs
        total_loc_loss += float(loc_loss.item()) * bs
        total_n += bs
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        if pos_mask.any():
            mask_np = pos_mask.detach().cpu().numpy()
            all_loc_pred.append(loc_pred.detach().cpu().numpy()[mask_np])
            all_loc_gt.append(gt_sample.detach().cpu().numpy()[mask_np])

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = cls_metrics(y_true, y_pred, class_names)
    if all_loc_gt:
        loc_pred_np = np.concatenate(all_loc_pred)
        loc_gt_np = np.concatenate(all_loc_gt)
    else:
        loc_pred_np = np.array([], dtype=np.int64)
        loc_gt_np = np.array([], dtype=np.int64)
    metrics.update(loc_metrics(loc_pred_np, loc_gt_np, cfg.hit_tol))

    losses = {
        "loss": total_loss / max(total_n, 1),
        "cls_loss": total_cls_loss / max(total_n, 1),
        "loc_loss": total_loc_loss / max(total_n, 1),
    }
    return losses, metrics


def save_history(history: list[dict], run_dir: Path, hit_tol: int) -> None:
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "history.csv", index=False)

    plt.figure(figsize=(8, 5))
    for col in ["train_loss", "val_loss", "train_cls_loss", "val_cls_loss", "train_loc_loss", "val_loc_loss"]:
        if col in df:
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for col in ["val_accuracy", "val_f1", f"val_loc_hit@{hit_tol}"]:
        if col in df:
            plt.plot(df["epoch"], df[col], label=col)
    if "val_loc_mae" in df:
        plt.plot(df["epoch"], df["val_loc_mae"] / max(float(df["val_loc_mae"].max()), 1.0), label="val_loc_mae_norm")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "metrics.png", dpi=200)
    plt.close()


def run_training(cfg: TrainConfig) -> dict:
    if cfg.task != "multiclass6":
        raise ValueError("This script is intended for multiclass6 + synchronization training.")
    if cfg.model_name not in SYNC_MODELS:
        raise ValueError(f"Choose a model with loc_head: {sorted(SYNC_MODELS)}")

    set_seed(cfg.seed)
    num_classes = task_num_classes(cfg.task)  # type: ignore[arg-type]
    run_name = f"{cfg.task}_sync_{cfg.model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_loader = build_loader(cfg.train_csv, cfg, shuffle=True)
    val_loader = build_loader(cfg.val_csv, cfg, shuffle=False)
    model = build_model(cfg.model_name, num_classes=num_classes).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score = -math.inf
    stale = 0
    history: list[dict] = []
    best_path = run_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")
        train_losses, train_metrics = run_epoch(model, train_loader, cfg, optimizer)
        val_losses, val_metrics = run_epoch(model, val_loader, cfg, None)

        loc_mae = val_metrics["loc_mae"]
        loc_hit = val_metrics[f"loc_hit@{cfg.hit_tol}"]
        loc_penalty = 0.0 if math.isnan(float(loc_mae)) else cfg.score_loc_weight * float(loc_mae)
        if cfg.select_by == "joint":
            score = float(val_metrics["f1"]) - loc_penalty
        elif cfg.select_by == "f1":
            score = float(val_metrics["f1"])
        elif cfg.select_by == "loc_mae":
            score = -float(loc_mae) if not math.isnan(float(loc_mae)) else -math.inf
        elif cfg.select_by == "loc_hit":
            score = float(loc_hit) if not math.isnan(float(loc_hit)) else -math.inf
        else:
            raise ValueError(f"Unknown select_by: {cfg.select_by}")

        row = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "train_cls_loss": train_losses["cls_loss"],
            "train_loc_loss": train_losses["loc_loss"],
            "val_loss": val_losses["loss"],
            "val_cls_loss": val_losses["cls_loss"],
            "val_loc_loss": val_losses["loc_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_loc_mae": val_metrics["loc_mae"],
            "val_loc_rmse": val_metrics["loc_rmse"],
            f"val_loc_hit@{cfg.hit_tol}": val_metrics[f"loc_hit@{cfg.hit_tol}"],
            "val_score": score,
        }
        history.append(row)
        save_history(history, run_dir, cfg.hit_tol)
        print(
            f"train_loss={row['train_loss']:.5f} val_loss={row['val_loss']:.5f} "
            f"val_f1={row['val_f1']:.5f} loc_mae={row['val_loc_mae']:.3f} "
            f"loc_hit@{cfg.hit_tol}={row[f'val_loc_hit@{cfg.hit_tol}']:.4f} score={score:.5f}"
        )

        if score - best_score > cfg.early_stop_min_delta:
            best_score = score
            stale = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "model_name": cfg.model_name,
                "task": cfg.task,
                "num_classes": num_classes,
                "class_names": task_class_names(cfg.task),  # type: ignore[arg-type]
                "epoch": epoch,
                "best_score": best_score,
                "val_metrics": val_metrics,
                "val_losses": val_losses,
                "has_sync_head": True,
            }
            torch.save(ckpt, best_path)
            (run_dir / "best_metrics.json").write_text(json.dumps(ckpt["val_metrics"], indent=2), encoding="utf-8")
            print(f"Saved best checkpoint: {best_path}")
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print(f"Early stop after {stale} stale epochs.")
                break

    result = {"run_dir": str(run_dir), "best_path": str(best_path), "best_score": best_score}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=sorted(SYNC_MODELS), default=TrainConfig.model_name)
    parser.add_argument("--data-root", default=TrainConfig.data_root)
    parser.add_argument("--train-csv", default=TrainConfig.train_csv)
    parser.add_argument("--val-csv", default=TrainConfig.val_csv)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--lambda-cls", type=float, default=TrainConfig.lambda_cls)
    parser.add_argument("--lambda-loc", type=float, default=TrainConfig.lambda_loc)
    parser.add_argument("--triangular-half-width", type=int, default=TrainConfig.triangular_half_width)
    parser.add_argument("--hit-tol", type=int, default=TrainConfig.hit_tol)
    parser.add_argument("--score-loc-weight", type=float, default=TrainConfig.score_loc_weight)
    parser.add_argument("--select-by", choices=["joint", "f1", "loc_mae", "loc_hit"], default=TrainConfig.select_by)
    parser.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default=TrainConfig.device)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    result = run_training(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
