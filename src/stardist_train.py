"""
Train a StarDist2D model or a fallback UNet model for instance segmentation.

This script provides a comprehensive training pipeline for segmenting objects
like alpha granules in electron microscopy images. It supports training with
the StarDist library or, if unavailable, a simple PyTorch UNet.

Key Features:
- Two modes: StarDist (recommended) or UNet (fallback).
- Data loading from a manifest CSV via `src.dataset.TileDataset`.
- Augmentation using `src.augment`.
- Mixed-precision training on GPU.
- Logging of train/validation loss, IoU, and Panoptic Quality (PQ).
- Checkpointing based on the best validation Panoptic Quality.
- Deterministic training via a random seed.

Example Command:
----------------
# Train a StarDist model with augmentations on GPU 0
python src/stardist_train.py \
    --manifest /path/to/your/dataset/manifest.csv \
    --checkpoint_dir experiments/stardist_model \
    --gpu 0 \
    --epochs 300 \
    --batch_size 4 \
    --lr 3e-4 \
    --patch_size 512 \
    --use_pretrained \
    --augment \
    --seed 42

# Train a UNet model on CPU if stardist is not installed
python src/stardist_train.py \
    --manifest /path/to/your/dataset/manifest.csv \
    --checkpoint_dir experiments/unet_model \
    --gpu -1 \
    --epochs 100 \
    --batch_size 8 \
    --augment
----------------

Hyperparameter Tips for Small Datasets:
- **Transfer Learning (`--use_pretrained`):** Highly recommended. It initializes the
  backbone with weights pre-trained on a large, diverse dataset, which helps
  the model learn relevant features faster and with less data.
- **Heavy Augmentation (`--augment`):** Essential. It artificially expands the
  dataset by creating modified versions of existing images, making the model
  more robust to variations in rotation, scale, brightness, etc.
- **Learning Rate (`--lr`):** A smaller learning rate (e.g., 1e-4 to 5e-5) is
  often better for fine-tuning pre-trained models.
- **Pseudo-labeling (Advanced):** If you have a lot of unlabeled data, you can
  use a trained model to make predictions, and then add the most confident
  predictions back into the training set as new labeled data. This is an

  advanced technique not implemented here but can be very effective.
"""
import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Dependency Imports and Fallbacks ---
try:
    from stardist.models import Config2D, StarDist2D
    from stardist.matching import matching_dataset
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False
    # Define dummy classes for type hinting and structure if stardist is missing
    class Config2D:
        pass
    class StarDist2D:
        pass
    class matching_dataset:
        pass


# Local module imports
from src.dataset import TileDataset
from src.augment import get_train_transforms, get_val_transforms


# --- Fallback UNet Model ---
class UNet(nn.Module):
    """A simple U-Net for binary segmentation as a fallback."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        def down(in_c, out_c):
            return nn.Sequential(nn.MaxPool2d(2), double_conv(in_c, out_c))

        class up(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
                self.conv = double_conv(in_c, out_c)
            def forward(self, x1, x2):
                x1 = self.up(x1)
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        self.inc = double_conv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def predict_instances(self, x, threshold=0.5):
        """Dummy function to mimic StarDist API for UNet."""
        from scipy.ndimage import label
        with torch.no_grad():
            logits = self.forward(x)
            pred = torch.sigmoid(logits)
            binary_pred = (pred > threshold).cpu().numpy().squeeze()
            labeled_pred, _ = label(binary_pred)
            return labeled_pred, {'coord': None} # Dummy coord


# --- Utility Functions ---

def set_seed(seed: int):
    """Sets a random seed for deterministic experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(gpu_id: int) -> torch.device:
    """Gets the torch device based on GPU ID."""
    if gpu_id >= 0 and torch.cuda.is_available():
        print(f"Using GPU: {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    print("Using CPU")
    return torch.device("cpu")

def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """Calculates segmentation metrics using stardist.matching."""
    if not STARDIST_AVAILABLE:
        print("Warning: StarDist not available, cannot compute PQ and IoU metrics.")
        return {"pq": 0, "iou": 0}
    
    try:
        stats = matching_dataset(y_true, y_pred, thresh=0.5, show_progress=False)
        return {"pq": stats.panoptic_quality, "iou": stats.mean_true_score}
    except Exception as e:
        print(f"Error calculating metrics: {e}. Returning 0.")
        return {"pq": 0, "iou": 0}


# --- Main Training Function ---

def main(args):
    """Main function to run the training and validation pipeline."""
    if args.seed is not None:
        set_seed(args.seed)

    # --- Setup ---
    device = get_device(args.gpu)
    log_dir = os.path.join(os.path.dirname(args.checkpoint_dir), "logs")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.csv")

    # --- Data Loading ---
    print("Loading data...")
    manifest = pd.read_csv(args.manifest)
    # Simple split: first 80% for train, last 20% for val.
    # A 'split' column in the manifest is a more robust approach.
    train_df = manifest.sample(frac=0.8, random_state=args.seed)
    val_df = manifest.drop(train_df.index)

    train_df.to_csv(os.path.join(log_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(log_dir, 'val_split.csv'), index=False)

    train_transform = get_train_transforms(args.patch_size) if args.augment else get_val_transforms(args.patch_size)
    val_transform = get_val_transforms(args.patch_size)

    train_dataset = TileDataset(os.path.join(log_dir, 'train_split.csv'), transform=train_transform)
    val_dataset = TileDataset(os.path.join(log_dir, 'val_split.csv'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()//2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=os.cpu_count()//2)

    # --- Model Initialization ---
    print("Initializing model...")
    use_stardist = STARDIST_AVAILABLE
    
    if use_stardist:
        print("Using StarDist2D model.")
        n_rays = 32
        # Use OpenCL-based computations for 'dist'/'dist_to_coord' if available
        use_gpu = device.type == 'cuda' and torch.cuda.is_available()
        
        conf = Config2D(
            n_rays=n_rays,
            grid=(1, 1),
            use_gpu=use_gpu,
            n_channel_in=1,
            train_patch_size=(args.patch_size, args.patch_size),
            train_batch_size=args.batch_size,
        )
        if args.use_pretrained:
            model = StarDist2D.from_pretrained('2D_versatile_fluo', config=conf, name='stardist', basedir=args.checkpoint_dir)
        else:
            model = StarDist2D(conf, name='stardist', basedir=args.checkpoint_dir)
        
        # Access the underlying torch model to train in a standard loop
        torch_model = model.net.to(device)
        criterion = model.criterion
    else:
        print("Warning: StarDist not found. Falling back to simple UNet.")
        torch_model = UNet(in_channels=1, out_channels=1).to(device)
        criterion = nn.BCEWithLogitsLoss()

    # --- Training Setup ---
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    best_val_pq = -1.0
    log_data = []

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # --- Training Phase ---
        torch_model.train()
        epoch_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = torch_model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_losses.append(loss.item())
            train_pbar.set_postfix(loss=loss.item())
        
        train_loss = np.mean(epoch_losses)

        # --- Validation Phase ---
        torch_model.eval()
        val_losses = []
        y_true_val, y_pred_val = [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")

        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                # Calculate loss
                outputs = torch_model(images)
                val_loss = criterion(outputs, masks)
                val_losses.append(val_loss.item())

                # Generate predictions for metrics
                for i in range(images.shape[0]):
                    true_mask = masks[i].cpu().numpy().astype(np.uint16)
                    
                    if use_stardist:
                        # StarDist needs single image prediction
                        pred_mask, _ = model._predict_instances_generator(images[i:i+1])
                    else:
                        # UNet fallback prediction
                        pred_mask = torch_model.predict_instances(images[i:i+1])

                    y_true_val.append(true_mask)
                    y_pred_val.append(pred_mask.astype(np.uint16))
                
                val_pbar.set_postfix(loss=val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        val_metrics = calculate_metrics(y_true_val, y_pred_val)
        
        # --- Logging & Checkpointing ---
        epoch_end_time = time.time()
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': avg_val_loss,
            'val_pq': val_metrics['pq'],
            'val_iou': val_metrics['iou'],
            'timestamp': epoch_end_time
        }
        log_data.append(log_entry)
        pd.DataFrame(log_data).to_csv(log_file, index=False)

        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PQ: {val_metrics['pq']:.4f}, Val IoU: {val_metrics['iou']:.4f}")

        if val_metrics['pq'] > best_val_pq:
            best_val_pq = val_metrics['pq']
            print(f"New best model found! Saving checkpoint to {args.checkpoint_dir}")
            
            # Save standard PyTorch checkpoint
            torch.save(torch_model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            
            if use_stardist:
                # StarDist has its own export method for deployment
                try:
                    model.export_TF(os.path.join(args.checkpoint_dir, 'export'))
                    print("StarDist model exported for deployment.")
                except Exception as e:
                    print(f"Could not export StarDist model: {e}")

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StarDist / UNet Training Script")
    parser.add_argument('--manifest', type=str, required=True, help='Path to the dataset manifest CSV file.')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--patch_size', type=int, default=512, help='Size of the image patches.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (-1 for CPU).')
    parser.add_argument('--use_pretrained', action='store_true', help='Use a pre-trained StarDist model as a starting point.')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    
    # Check if we should even try to use stardist
    if not STARDIST_AVAILABLE and args.use_pretrained:
        print("Warning: --use_pretrained is set, but stardist library is not available. Will train a UNet from scratch.")

    main(args)
