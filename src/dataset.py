import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================================
# DATASET BUILDER
# ============================================================

def build_manifest(
    image_dir: Path,
    annotation_dir: Path,
    out_dir: Path,
    tile_size: int = 512,
    overlap: int = 64,
    save_empty: bool = False,
):
    """
    Builds a tiled dataset and writes a manifest.csv.
    """

    images_out = out_dir / "images"
    masks_out = out_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    image_files = sorted(image_dir.glob("*"))

    for img_path in tqdm(image_files, desc="Processing images"):
        ann_path = annotation_dir / img_path.name
        if not ann_path.exists():
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ann = cv2.imread(str(ann_path))

        if image is None or ann is None:
            continue

        # ====================================================
        # HSV-BASED COLOR EXTRACTION
        # ====================================================

        hsv = cv2.cvtColor(ann, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)

        blue_mask = ((h >= 85) & (h <= 150) & (s >= 30))
        yellow_mask = ((h >= 20) & (h <= 45) & (s >= 30))

        outline = (blue_mask & (~yellow_mask)).astype(np.uint8)

        # ====================================================
        # GAP CLOSING (STRONG ENOUGH FOR FILL)
        # ====================================================

        kernel = np.ones((3, 3), np.uint8)
        outline = cv2.morphologyEx(
            outline, cv2.MORPH_CLOSE, kernel, iterations=3
        )

        # ====================================================
        # FLOOD FILL (PRIMARY SEGMENTATION)
        # ====================================================

        inv = 1 - outline
        h_, w_ = inv.shape

        floodfill = inv.copy()
        ff_mask = np.zeros((h_ + 2, w_ + 2), np.uint8)
        cv2.floodFill(floodfill, ff_mask, (0, 0), 2)

        mask = (floodfill != 2).astype(np.uint8)
        filled_area = mask.sum()

        # ====================================================
        # BIOLOGICAL CLEANUP (SOFT, SAFE)
        # ====================================================

        num_labels, labels = cv2.connectedComponents(mask)
        clean_mask = np.zeros_like(mask)

        MIN_AREA = 30
        MAX_AREA = 2000
        MAX_ASPECT = 4.0

        for label_id in range(1, num_labels):
            component = (labels == label_id)
            area = component.sum()

            if area < MIN_AREA or area > MAX_AREA:
                continue

            ys, xs = np.where(component)
            h_box = ys.max() - ys.min() + 1
            w_box = xs.max() - xs.min() + 1
            aspect_ratio = max(
                h_box / (w_box + 1e-6),
                w_box / (h_box + 1e-6)
            )

            if aspect_ratio > MAX_ASPECT:
                continue

            clean_mask[component] = 1

        # SAFETY: do not destroy valid fills
        if clean_mask.sum() >= 0.5 * filled_area:
            mask = clean_mask
        # else: keep original filled mask

        # ====================================================
        # TILING
        # ====================================================

        H, W = image.shape
        stride = tile_size - overlap
        tile_id = 0

        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                img_tile = image[y:y + tile_size, x:x + tile_size]
                mask_tile = mask[y:y + tile_size, x:x + tile_size]

                if not save_empty and mask_tile.sum() == 0:
                    continue

                tile_name = f"{img_path.stem}_tile_{tile_id:06d}.png"

                cv2.imwrite(str(images_out / tile_name), img_tile)
                cv2.imwrite(str(masks_out / tile_name), mask_tile * 255)

                manifest_rows.append({
                    "image_path": str(images_out / tile_name),
                    "mask_path": str(masks_out / tile_name),
                })

                tile_id += 1

    df = pd.DataFrame(manifest_rows)
    df.to_csv(out_dir / "manifest.csv", index=False)

    print(f"Saved {len(df)} tiles → {out_dir / 'manifest.csv'}")


# ============================================================
# PYTORCH DATASET
# ============================================================

class TileDataset(Dataset):
    def __init__(self, manifest_path: str, transform: Optional[Callable] = None):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        self.manifest_df = pd.read_csv(manifest_path)
        self.transform = transform

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]

        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

        mask = (mask > 0).astype(np.uint16)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()

        return image, mask
