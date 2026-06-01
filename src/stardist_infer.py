"""
Inference script for StarDist-2D and fallback UNet models.

This script takes a trained model and applies it to one or more images to
perform instance segmentation. It is designed to handle large images by
tiling them, performing inference on each tile, and stitching the results.

Key Features:
- Supports both StarDist models and simple PyTorch UNet models (.pth).
- Handles large images via tiled inference with configurable overlap.
- Resolves duplicate predictions in overlapping regions using NMS (for StarDist)
  or heatmap averaging (for UNet).
- Filters small, spurious objects based on a minimum area threshold.
- Outputs rich information for each input image:
  1. An overlay PNG visualizing the detected instances.
  2. A CSV file with detailed statistics for each instance (area, centroid, etc.).
  3. An optional raw label image in .npy format.
- Reports inference speed in tiles/second.

Example Usage:
----------------
# Using a trained StarDist model on a folder of images
python src/stardist_infer.py \
    --model experiments/stardist_model \
    --input /path/to/images \
    --out_dir experiments/inference_results \
    --tile_size 1024 \
    --overlap 128 \
    --min_area_px 50 \
    --device cuda:0

# Using a trained UNet model (.pth) on a single large image
python src/stardist_infer.py \
    --model experiments/unet_model/best_model.pth \
    --input /path/to/large_image.tif \
    --out_dir experiments/inference_results_unet \
    --tile_size 512 \
    --overlap 64 \
    --save_npy
----------------
"""
import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import label
from skimage.measure import regionprops_table
from skimage.segmentation import watershed
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
# This must match the definition used during UNet training.
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
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# --- Core Functions ---

def load_model(model_path: str, device: torch.device) -> tuple[nn.Module | StarDist2D, str]:
    """Loads a StarDist or UNet model."""
    if os.path.isdir(model_path) and STARDIST_AVAILABLE:
        print(f"Loading StarDist model from: {model_path}")
        # StarDist model is a directory, load it by providing the name and base directory
        model = StarDist2D(config=None, name=os.path.basename(model_path), basedir=os.path.dirname(model_path))
        if model.config.use_gpu:
            model.net.to(device)
        return model, "stardist"
    elif os.path.isfile(model_path) and model_path.endswith('.pth'):
        print(f"Loading UNet model from: {model_path}")
        model = UNet(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, "unet"
    else:
        raise ValueError(f"Model path not recognized: {model_path}. "
                         "Provide a directory for a StarDist model or a .pth file for a UNet.")

def calculate_instance_stats(label_image: np.ndarray, intensity_image: np.ndarray) -> pd.DataFrame:
    """Calculates statistics for each labeled instance."""
    if label_image.max() == 0:
        return pd.DataFrame() # No instances found

    props = regionprops_table(
        label_image,
        intensity_image=intensity_image,
        properties=('label', 'centroid', 'area', 'mean_intensity', 'bbox')
    )
    df = pd.DataFrame(props)
    # Rename columns for clarity
    df.rename(columns={
        'label': 'instance_id',
        'centroid-0': 'centroid_y',
        'centroid-1': 'centroid_x',
        'area': 'area_px',
        'bbox-0': 'bbox_y1',
        'bbox-1': 'bbox_x1',
        'bbox-2': 'bbox_y2',
        'bbox-3': 'bbox_x2',
    }, inplace=True)
    return df

def filter_small_instances(label_image: np.ndarray, min_area_px: int) -> np.ndarray:
    """Removes instances smaller than a given area threshold."""
    if min_area_px <= 0:
        return label_image
    
    component_sizes = np.bincount(label_image.ravel())
    too_small = component_sizes < min_area_px
    too_small_mask = too_small[label_image]
    label_image[too_small_mask] = 0
    return label_image

# --- Main Inference Logic ---

def main(args):
    """Main function to run the inference pipeline."""
    # --- Setup ---
    device = torch.device(args.device)
    overlay_dir = os.path.join(args.out_dir, "overlays")
    csv_dir = os.path.join(args.out_dir, "csv")
    npy_dir = os.path.join(args.out_dir, "npy")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    if args.save_npy:
        os.makedirs(npy_dir, exist_ok=True)

    model, model_type = load_model(args.model, device)

    # --- Find Input Images ---
    if os.path.isdir(args.input):
        image_files = sorted(list(Path(args.input).glob('*.*')))
    else:
        image_files = [Path(args.input)]
    
    print(f"Found {len(image_files)} image(s) to process.")

    # --- Process Each Image ---
    for img_path in tqdm(image_files, desc="Processing Images"):
        base_name = img_path.stem
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        
        start_time = time.time()
        
        if model_type == "stardist":
            # StarDist has a built-in function for tiled prediction
            labels, _ = model.predict_instances_big(
                image,
                axes='YX',
                block_size=args.tile_size,
                min_overlap=args.overlap,
                context=args.overlap // 2,
                show_progress=False
            )
            num_tiles = (image.shape[0] // (args.tile_size - args.overlap) + 1) * \
                      (image.shape[1] // (args.tile_size - args.overlap) + 1)

        else: # UNet Fallback
            # Manual tiling, stitching, and watershed
            h, w = image.shape
            pad_h = (args.tile_size - h % args.tile_size) % args.tile_size
            pad_w = (args.tile_size - w % args.tile_size) % args.tile_size
            padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            stitched_heatmap = np.zeros(padded_image.shape, dtype=np.float32)
            norm_mask = np.zeros(padded_image.shape, dtype=np.float32)
            
            y_steps = range(0, padded_image.shape[0], args.tile_size - args.overlap)
            x_steps = range(0, padded_image.shape[1], args.tile_size - args.overlap)
            
            num_tiles = 0
            for y in y_steps:
                for x in x_steps:
                    if y + args.tile_size > padded_image.shape[0] or x + args.tile_size > padded_image.shape[1]:
                        continue
                    tile = padded_image[y:y+args.tile_size, x:x+args.tile_size]
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
                    
                    with torch.no_grad():
                        output = model(tile_tensor)
                    
                    heatmap = torch.sigmoid(output).squeeze().cpu().numpy()
                    stitched_heatmap[y:y+args.tile_size, x:x+args.tile_size] += heatmap
                    norm_mask[y:y+args.tile_size, x:x+args.tile_size] += 1
                    num_tiles += 1

            stitched_heatmap /= norm_mask
            stitched_heatmap = stitched_heatmap[:h, :w] # Crop back to original size

            # Watershed to get instances
            thresholded = stitched_heatmap > 0.5
            markers, _ = label(thresholded)
            labels = watershed(-stitched_heatmap, markers, mask=thresholded)

        # --- Post-processing ---
        if args.min_area_px > 0:
            labels = filter_small_instances(labels, args.min_area_px)
        
        end_time = time.time()
        
        # --- Save Outputs ---
        stats_df = calculate_instance_stats(labels, image)
        stats_df.to_csv(os.path.join(csv_dir, f"{base_name}_stats.csv"), index=False)

        if args.save_npy:
            np.save(os.path.join(npy_dir, f"{base_name}_labels.npy"), labels)

        # Visualization
        if STARDIST_AVAILABLE:
            # Normalize image for visualization
            img_norm = (image - image.min()) / (image.max() - image.min())
            img_norm = (img_norm * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
            overlay = render_label(labels, img=img_rgb, alpha=0.5)
            cv2.imwrite(os.path.join(overlay_dir, f"{base_name}_overlay.png"), overlay)
        else:
            print("Skipping overlay: StarDist not available for `render_label` function.")

        # --- Report Performance ---
        duration = end_time - start_time
        tiles_per_sec = num_tiles / duration if duration > 0 else float('inf')
        print(f"Processed {base_name}: {stats_df.shape[0]} instances found. "
              f"({num_tiles} tiles @ {tiles_per_sec:.2f} tiles/sec)")

        # GPU Memory Management
        if 'cuda' in args.device:
            torch.cuda.empty_cache()
    
    print("\nInference complete.")
    print(f"Results saved in: {args.out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StarDist/UNet Inference Script")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model directory (StarDist) or .pth file (UNet).')
    parser.add_argument('--input', type=str, required=True, help='Path to a single input image or a directory of images.')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save all output files.')
    parser.add_argument('--tile_size', type=int, default=1024, help='The size of tiles for processing large images.')
    parser.add_argument('--overlap', type=int, default=128, help='Pixel overlap between adjacent tiles.')
    parser.add_argument('--min_area_px', type=int, default=50, help='Minimum area in pixels to keep a detected instance.')
    parser.add_argument('--save_npy', action='store_true', help='If set, saves the final label image as a .npy file.')
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for inference, e.g., 'cuda:0' or 'cpu'.")

    args = parser.parse_args()
    main(args)
