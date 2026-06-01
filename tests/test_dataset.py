import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.dataset import TileDataset


def test_tile_dataset_creates_and_reads_manifest(tmp_path):
    """Create a small synthetic image and mask, write manifest, and verify TileDataset."""
    data_dir = tmp_path / "data"
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    data_dir.mkdir()
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create a simple 64x64 grayscale image
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (20, 20), 8, 200, -1)
    cv2.circle(img, (45, 40), 10, 150, -1)

    # Create an instance-labeled mask (uint16) with two objects
    mask = np.zeros((64, 64), dtype=np.uint16)
    cv2.circle(mask, (20, 20), 8, 1, -1)
    cv2.circle(mask, (45, 40), 10, 2, -1)

    img_path = img_dir / "sample.png"
    mask_path = mask_dir / "sample.png"
    cv2.imwrite(str(img_path), img)
    # Write mask as PNG; cv2 will preserve integer values for small labels
    cv2.imwrite(str(mask_path), mask)

    # Create manifest CSV
    manifest_path = data_dir / "manifest.csv"
    df = pd.DataFrame({
        'image_path': [str(img_path)],
        'mask_path': [str(mask_path)],
    })
    df.to_csv(manifest_path, index=False)

    # Instantiate dataset and read a sample
    ds = TileDataset(manifest_path=str(manifest_path))
    assert len(ds) == 1

    image_tensor, mask_tensor = ds[0]

    # Image should be a torch Tensor with shape (1, H, W) and float dtype
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.ndim == 3 and image_tensor.shape[0] == 1

    # Mask should be a torch Tensor (long) with same H,W
    assert isinstance(mask_tensor, torch.Tensor)
    assert mask_tensor.ndim == 2

    # Ensure mask contains the two instance labels
    unique = torch.unique(mask_tensor)
    # background 0 plus labels 1 and 2
    assert set(unique.tolist()) >= {0, 1, 2}

