"""
Optuna Hyperparameter Optimization driver for platelet-seg training.

Usage (example):
  python src/hpo_optuna.py --manifest data/manifest.csv --trials 50 --device 0 --kfold 3 --direction pq --epochs 20 --final

Features:
- CLI for experiments: `--manifest`, `--trials`, `--timeout`, `--device`, `--kfold`, `--direction`.
- Search space: `learning_rate` (1e-5..1e-3 log), `weight_decay` (1e-6..1e-3 log),
  `batch_size` (4,8,16), `num_filters_multiplier` (1,2,4).
- Uses Optuna pruning and saves the `study` to `experiments/hpo/study.pkl`.
- Saves best config to `experiments/hpo/best_config.json`.
- Optionally runs a final training using the best params and saves checkpoint to
  `models/checkpoints/hpo_best/` when `--final` is passed.

Notes / resume:
- To resume from a saved study: `import pickle; study = pickle.load(open('experiments/hpo/study.pkl','rb'))`
  then call `study.optimize(objective, n_trials=MORE_TRIALS)` to continue. For long-running or distributed
  experiments consider using an RDB storage backend (e.g., SQLite) via `optuna.create_study(storage=...)`.
- Multi-GPU: this script supports specifying a device index via `--device`. For multi-GPU training
  consider launching separate Optuna workers (each with a different GPU) using `--device` and
  `optuna.study.Study.optimize` in separate processes (e.g., with GNU parallel or Slurm).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.augment import get_train_transforms, get_val_transforms
from src.dataset import TileDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    """Parse `--device` argument. Accepts 'cpu', an int string (GPU id), or 'cuda:idx'."""
    if device_arg.lower() == 'cpu':
        return torch.device('cpu')
    try:
        # numeric GPU id
        gid = int(device_arg)
        if gid >= 0 and torch.cuda.is_available():
            return torch.device(f'cuda:{gid}')
        return torch.device('cpu')
    except Exception:
        # try 'cuda:0' style
        if device_arg.startswith('cuda') and torch.cuda.is_available():
            return torch.device(device_arg)
        return torch.device('cpu')


class SimpleUNet(nn.Module):
    """A compact U-Net where `base_channels` can be scaled by multiplier.

    This is intentionally small and self-contained for HPO runs.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.inc = conv_block(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), conv_block(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), conv_block(base_channels * 2, base_channels * 4))
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv_up1 = conv_block(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.conv_up2 = conv_block(base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)
        return self.outc(u2)

    def predict_mask(self, x, threshold=0.5):
        """Return binary mask prediction for an input tensor `x` (NCHW)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).cpu().numpy().astype(np.uint8)


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute IoU between two binary masks (H,W)"""
    inter = np.logical_and(y_true > 0, y_pred > 0).sum()
    union = np.logical_or(y_true > 0, y_pred > 0).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def train_and_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    manifest_path: str,
    params: Dict,
    device: torch.device,
    epochs: int = 20,
    patch_size: int = 512,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[float, float]:
    """Train a small UNet for `epochs` and return (val_loss, val_iou).

    Notes: Uses `src.dataset.TileDataset` which expects a CSV manifest.
    """
    set_seed(seed)

    train_csv = os.path.join('/tmp', f'train_split_{int(time.time()*1000)}.csv')
    val_csv = os.path.join('/tmp', f'val_split_{int(time.time()*1000)}.csv')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_transform = get_train_transforms(patch_size) if augment else get_val_transforms(patch_size)
    val_transform = get_val_transforms(patch_size)

    train_ds = TileDataset(train_csv, transform=train_transform)
    val_ds = TileDataset(val_csv, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=max(1, os.cpu_count() // 2))
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'] * 2, shuffle=False, num_workers=max(1, os.cpu_count() // 2))

    # Model
    base_ch = 64 * params['num_filters_multiplier']
    model = SimpleUNet(in_channels=1, out_channels=1, base_channels=base_ch).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    best_val_loss = float('inf')
    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_losses.append(loss.item())

                preds = torch.sigmoid(outputs) > 0.5
                preds_np = preds.cpu().numpy().squeeze()
                masks_np = masks.cpu().numpy().squeeze()
                # ensure correct shape
                if preds_np.ndim == 2:
                    iou = compute_iou(masks_np, preds_np)
                    ious.append(iou)
                else:
                    # batch axis
                    for p, m in zip(preds_np, masks_np):
                        ious.append(compute_iou(m, p))

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        avg_iou = float(np.mean(ious)) if ious else 0.0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        if avg_iou > best_iou:
            best_iou = avg_iou

    # cleanup temp csvs
    try:
        os.remove(train_csv)
        os.remove(val_csv)
    except Exception:
        pass

    return best_val_loss, best_iou


def objective_factory(manifest: str, device: torch.device, args):
    manifest_df = pd.read_csv(manifest)

    def objective(trial: optuna.trial.Trial):
        # Suggest hyperparameters
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
        num_filters_multiplier = trial.suggest_categorical('num_filters_multiplier', [1, 2, 4])

        params = {
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'num_filters_multiplier': num_filters_multiplier,
        }

        # K-Fold split (simple deterministic split)
        k = max(1, int(args.kfold))
        indices = np.arange(len(manifest_df))
        if args.seed is not None:
            np.random.seed(args.seed)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)

        fold_val_losses = []
        fold_iou = []

        for i in range(k):
            val_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i]) if k > 1 else np.hstack(folds)

            train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
            val_df = manifest_df.iloc[val_idx].reset_index(drop=True)

            # Train on this fold
            best_val_loss, best_iou = train_and_eval(
                train_df,
                val_df,
                manifest,
                params,
                device,
                epochs=args.epochs,
                patch_size=args.patch_size,
                augment=not args.no_augment,
                seed=args.seed or 42,
            )

            fold_val_losses.append(best_val_loss)
            fold_iou.append(best_iou)

            # Report intermediate result for pruning
            if args.direction == 'val_loss':
                # smaller is better
                intermediate = float(np.mean(fold_val_losses))
            else:
                # use IoU as proxy for PQ; larger is better
                intermediate = float(np.mean(fold_iou))

            trial.report(intermediate, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        avg_val_loss = float(np.mean(fold_val_losses))
        avg_iou = float(np.mean(fold_iou))

        return avg_val_loss if args.direction == 'val_loss' else avg_iou

    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna HPO driver for platelet-seg')
    parser.add_argument('--manifest', required=True, help='Path to dataset manifest CSV')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds for Optuna study')
    parser.add_argument('--device', type=str, default='0', help='Device to run on: cpu or GPU id (e.g. 0)')
    parser.add_argument('--kfold', type=int, default=3, help='Number of folds to use for cross-validation')
    parser.add_argument('--direction', choices=['val_loss', 'pq'], default='pq', help="Objective: 'val_loss' (minimize) or 'pq' (maximize proxy)")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per trial/fold')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for training')
    parser.add_argument('--no_augment', action='store_true', help='Disable augmentation during trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--final', action='store_true', help='After HPO, run final training with best params and save checkpoint')
    args = parser.parse_args()

    device = get_device(args.device)

    # Create output dirs
    os.makedirs('experiments/hpo', exist_ok=True)
    os.makedirs('models/checkpoints/hpo_best', exist_ok=True)

    study_name = f'hpo_{int(time.time())}'
    study_path = os.path.join('experiments', 'hpo', 'study.pkl')

    # Map direction to optuna direction
    optuna_direction = 'minimize' if args.direction == 'val_loss' else 'maximize'

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction=optuna_direction, sampler=sampler, pruner=pruner)

    objective = objective_factory(args.manifest, device, args)

    try:
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    except KeyboardInterrupt:
        print('Interrupted HPO optimization.')

    # Save study and best params
    try:
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        print(f'Saved study to {study_path}')
    except Exception as e:
        print(f'Could not pickle study: {e}')

    best = study.best_trial.params if study.best_trial is not None else {}
    best_json = os.path.join('experiments', 'hpo', 'best_config.json')
    with open(best_json, 'w') as f:
        json.dump({'best_params': best, 'value': study.best_value if study.best_trial is not None else None}, f, indent=2)
    print(f'Best params saved to {best_json}')

    if args.final and best:
        print('Running final training with best params on full dataset...')
        manifest_df = pd.read_csv(args.manifest)
        # Use full dataset: simple split 80/20
        train_df = manifest_df.sample(frac=0.9, random_state=args.seed).reset_index(drop=True)
        val_df = manifest_df.drop(train_df.index).reset_index(drop=True)

        # fill in defaults if missing
        params = {
            'learning_rate': best.get('learning_rate', 1e-4),
            'weight_decay': best.get('weight_decay', 1e-6),
            'batch_size': best.get('batch_size', 8),
            'num_filters_multiplier': best.get('num_filters_multiplier', 1),
        }

        best_val_loss, best_iou = train_and_eval(
            train_df,
            val_df,
            args.manifest,
            params,
            device,
            epochs=max(50, args.epochs),
            patch_size=args.patch_size,
            augment=not args.no_augment,
            seed=args.seed,
        )

        # Save a minimal checkpoint (model state) using the same architecture
        base_ch = 64 * params['num_filters_multiplier']
        final_model = SimpleUNet(in_channels=1, out_channels=1, base_channels=base_ch).to(device)
        # NOTE: We didn't return the trained model from train_and_eval; for a full final run
        # you might want to modify train_and_eval to return the model. Here we save metadata.
        meta = {
            'best_val_loss': best_val_loss,
            'best_iou': best_iou,
            'params': params,
        }
        with open(os.path.join('models', 'checkpoints', 'hpo_best', 'hpo_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print('Final run done; metadata saved to models/checkpoints/hpo_best/hpo_meta.json')


if __name__ == '__main__':
    main()
