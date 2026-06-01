"""
Active Learning script to rank unlabeled images for annotation priority.

This script runs a trained model (StarDist or UNet) on a pool of unlabeled
images. It calculates various uncertainty metrics for each image and uses a
combined score to rank them. The top-ranked images are considered the most
informative for the model to learn from and should be prioritized for
manual annotation.

Uncertainty Metrics Calculated:
- **Image Entropy:** The entropy of the stitched prediction probability map.
  High entropy indicates the model is uncertain about large regions of the image.
- **Average Instance Uncertainty:** For each detected object, we find the model's
  peak confidence (max probability) within the object's mask. The uncertainty
  is (1 - confidence). This value is averaged over all objects in the image.
- **Proportion of Tiny Objects:** The fraction of detected objects that are
  smaller than a fixed pixel threshold. A high value may indicate noisy or
  fragmented predictions.

The final ranking is based on a weighted sum of these metrics, normalized
across the entire image pool.

Example Usage:
----------------
python src/active_learning.py \
    --model experiments/stardist_model \
    --pool_dir /path/to/unlabeled_images \
    --out_csv experiments/active_learning_candidates.csv \
    --top_k 20 \
    --tile_size 1024 \
    --overlap 128 \
    --device cuda:0
----------------
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import entropy
from skimage.measure import label, regionprops
from tqdm import tqdm

# --- Dependency Imports and Fallbacks ---
try:
    from stardist.models import StarDist2D
    from stardist.plot import render_label
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False
    class StarDist2D: pass
    def render_label(*args, **kwargs): pass

# --- Fallback UNet Model Definition ---
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
        self.inc = double_conv(in_channels, 64); self.down1 = down(64, 128); self.down2 = down(128, 256)
        self.down3 = down(256, 512); self.down4 = down(512, 1024); self.up1 = up(1024, 512)
        self.up2 = up(512, 256); self.up3 = up(256, 128); self.up4 = up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# --- Core Functions ---

def load_model(model_path: str, device: torch.device) -> tuple[nn.Module | StarDist2D, str]:
    """Loads a StarDist or UNet model."""
    if os.path.isdir(model_path) and STARDIST_AVAILABLE:
        model = StarDist2D(config=None, name=os.path.basename(model_path), basedir=os.path.dirname(model_path))
        if model.config.use_gpu: model.net.to(device)
        return model, "stardist"
    elif os.path.isfile(model_path) and model_path.endswith('.pth'):
        model = UNet(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, "unet"
    else:
        raise ValueError(f"Model path not recognized: {model_path}.")

def score_image(
    model: nn.Module | StarDist2D, model_type: str, image_path: str, device: torch.device, tile_size: int, overlap: int
) -> dict:
    """Runs inference on a single image and computes uncertainty scores."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return {}

    # --- Tiled Inference to get labels and probability map ---
    if model_type == "stardist":
        labels, details = model.predict_instances_big(
            image, axes='YX', block_size=tile_size, min_overlap=overlap,
            context=overlap // 2, show_progress=False, return_predict=True
        )
        prob_map = details['prob']
    else: # UNet
        h, w = image.shape
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        prob_map = np.zeros(padded_image.shape, dtype=np.float32)
        norm_mask = np.zeros(padded_image.shape, dtype=np.float32)
        
        y_steps = range(0, padded_image.shape[0], tile_size - overlap)
        x_steps = range(0, padded_image.shape[1], tile_size - overlap)
        
        for y in y_steps:
            for x in x_steps:
                if y + tile_size > padded_image.shape[0] or x + tile_size > padded_image.shape[1]: continue
                tile = padded_image[y:y+tile_size, x:x+tile_size]
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    output = model(tile_tensor)
                heatmap = torch.sigmoid(output).squeeze().cpu().numpy()
                prob_map[y:y+tile_size, x:x+tile_size] += heatmap
                norm_mask[y:y+tile_size, x:x+tile_size] += 1
        
        prob_map /= (norm_mask + 1e-8)
        prob_map = prob_map[:h, :w]
        labels = label(prob_map > 0.5)

    # --- Compute Scores ---
    num_instances = labels.max()
    image_entropy = entropy(prob_map.flatten()).item()

    avg_instance_uncertainty = 0.0
    tiny_instances = 0
    TINY_AREA_THRESH = 20

    if num_instances > 0:
        instance_uncertainties = []
        props = regionprops(labels)
        for prop in props:
            # Instance uncertainty
            instance_mask = (labels == prop.label)
            max_prob = np.max(prob_map[instance_mask])
            instance_uncertainties.append(1.0 - max_prob)
            # Tiny proportion
            if prop.area < TINY_AREA_THRESH:
                tiny_instances += 1
        
        avg_instance_uncertainty = np.mean(instance_uncertainties) if instance_uncertainties else 0
        prop_tiny = tiny_instances / num_instances
    else:
        prop_tiny = 0.0

    return {
        'image_path': image_path,
        'num_instances': num_instances,
        'image_entropy': image_entropy,
        'avg_instance_uncertainty': avg_instance_uncertainty,
        'prop_tiny': prop_tiny,
        '_labels_cache': labels, # Cache labels for visualization
    }

def rank_pool(args):
    """Main function to rank the image pool and save results."""
    # --- Setup ---
    device = torch.device(args.device)
    out_dir_viz = Path("experiments/active_candidates/")
    out_dir_viz.mkdir(exist_ok=True, parents=True)
    
    model, model_type = load_model(args.model, device)
    
    image_files = sorted(list(Path(args.pool_dir).glob('*.*')))
    if not image_files:
        print(f"No images found in --pool_dir: {args.pool_dir}")
        return

    # --- Score all images ---
    all_scores = []
    pbar = tqdm(image_files, desc="Scoring images")
    for img_path in pbar:
        scores = score_image(model, model_type, str(img_path), device, args.tile_size, args.overlap)
        if scores:
            all_scores.append(scores)
    
    if not all_scores:
        print("No images were successfully scored.")
        return

    # --- Rank based on combined score ---
    df = pd.DataFrame(all_scores)
    
    # Normalize score components to [0, 1] range
    df['entropy_norm'] = (df['image_entropy'] - df['image_entropy'].min()) / (df['image_entropy'].max() - df['image_entropy'].min() + 1e-8)
    df['inst_unc_norm'] = (df['avg_instance_uncertainty'] - df['avg_instance_uncertainty'].min()) / (df['avg_instance_uncertainty'].max() - df['avg_instance_uncertainty'].min() + 1e-8)
    df['tiny_norm'] = (df['prop_tiny'] - df['prop_tiny'].min()) / (df['prop_tiny'].max() - df['prop_tiny'].min() + 1e-8)
    
    # Combine scores (equal weighting)
    df['combined_score'] = df['entropy_norm'] + df['inst_unc_norm'] + df['tiny_norm']
    
    df_sorted = df.sort_values(by='combined_score', ascending=False)
    
    # --- Save Outputs ---
    top_k_df = df_sorted.head(args.top_k)
    
    # Save CSV
    out_csv_df = top_k_df[['image_path', 'combined_score', 'num_instances', 'avg_instance_uncertainty', 'image_entropy', 'prop_tiny']]
    out_csv_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved top {args.top_k} candidates to: {args.out_csv}")

    # Save visualization overlays for top_k
    print(f"Saving visualization for top {args.top_k} candidates to: {out_dir_viz}")
    for _, row in top_k_df.iterrows():
        base_name = Path(row['image_path']).stem
        labels = row['_labels_cache']
        
        # We need to re-read the original image for the background
        image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        
        if STARDIST_AVAILABLE:
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            img_rgb = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            overlay = render_label(labels, img=img_rgb, alpha=0.5)
            out_path = out_dir_viz / f"rank_{len(out_csv_df) - _.name}_score_{row['combined_score']:.2f}_{base_name}.png"
            cv2.imwrite(str(out_path), overlay)
        else:
            print("Skipping overlay generation: StarDist not available for visualization.")
            break # No need to repeat this message

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active Learning Ranking Script")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model directory (StarDist) or .pth file (UNet).')
    parser.add_argument('--pool_dir', type=str, required=True, help='Directory of unlabeled images to rank.')
    parser.add_argument('--out_csv', type=str, default='active_learning_candidates.csv', help='Path to save the output CSV of top candidates.')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top candidates to select.')
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for inference, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument('--tile_size', type=int, default=512, help='The size of tiles for processing large images.')
    parser.add_argument('--overlap', type=int, default=64, help='Pixel overlap between adjacent tiles.')

    args = parser.parse_args()
    rank_pool(args)
