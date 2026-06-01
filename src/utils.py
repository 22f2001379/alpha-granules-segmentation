"""
Utility functions for platelet-seg analysis and data handling.

This module provides helper functions for:
- Aggregating segmentation prediction statistics
- Filtering small objects from label images
- Unit conversions (pixel to micron)
- Manifest and JSON I/O
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage


def read_manifest(manifest_csv: str) -> List[Dict[str, Any]]:
    """Load dataset manifest CSV file into a list of dictionaries.

    Each row in the CSV becomes a dictionary with column names as keys.

    Args:
        manifest_csv: Path to the manifest CSV file.

    Returns:
        List of dictionaries, one per row.

    Example:
        >>> manifest = read_manifest('data/manifest.csv')
        >>> len(manifest)
        150
        >>> manifest[0].keys()
        dict_keys(['image_path', 'mask_path', 'split'])
    """
    df = pd.read_csv(manifest_csv)
    return df.to_dict('records')


def save_json(path: str, obj: Any) -> None:
    """Save a Python object to JSON file.

    Args:
        path: Destination file path.
        obj: Object to serialize (should be JSON-compatible).

    Example:
        >>> config = {'lr': 1e-4, 'batch_size': 8}
        >>> save_json('config.json', config)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    """Load a JSON file into a Python object.

    Args:
        path: Path to JSON file.

    Returns:
        Deserialized Python object.

    Example:
        >>> config = load_json('config.json')
        >>> config['lr']
        0.0001
    """
    with open(path, 'r') as f:
        return json.load(f)


def pixel_area_to_micron2(num_pixels: int, pixel_to_micron: float = 1.0) -> float:
    """Convert pixel area to square microns.

    Converts a pixel count to physical area in square microns.
    The conversion factor `pixel_to_micron` is the side length of one pixel in microns.

    Args:
        num_pixels: Number of pixels (area in pixel units).
        pixel_to_micron: Side length of one pixel in microns. Default: 1.0 (no conversion).

    Returns:
        Area in square microns (float).

    Notes:
        - If `pixel_to_micron` is not calibrated for your microscope/imaging setup,
          set it to 1.0 and interpret the result in pixel units.
        - For EM images at typical magnifications, pixel_to_micron might be ~0.001–0.1 µm.
        - Formula: area_micron2 = num_pixels * (pixel_to_micron ** 2)

    Example:
        >>> # 100 pixels with pixel_to_micron = 0.01 µm
        >>> pixel_area_to_micron2(100, pixel_to_micron=0.01)
        0.01
    """
    return float(num_pixels) * (pixel_to_micron ** 2)


def filter_small_objects(label_image: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Remove small connected components from a labeled image.

    Components with area < min_area are removed (set to 0). This is useful
    for cleaning up noise and artifacts in instance segmentation masks.

    Args:
        label_image: Integer array where each unique positive value represents
                     a connected component (e.g., output from scipy.ndimage.label).
        min_area: Minimum area (in pixels) to retain a component. Default: 10.

    Returns:
        Cleaned label image with small components removed.

    Example:
        >>> from scipy.ndimage import label
        >>> binary = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]])
        >>> labeled, _ = label(binary)
        >>> labeled
        array([[1, 1, 0],
               [1, 0, 2],
               [0, 0, 2]])
        >>> cleaned = filter_small_objects(labeled, min_area=2)
        >>> cleaned
        array([[1, 1, 0],
               [1, 0, 2],
               [0, 0, 2]])
        >>> cleaned = filter_small_objects(labeled, min_area=4)
        >>> cleaned  # Component 2 removed (only 2 pixels)
        array([[1, 1, 0],
               [1, 0, 0],
               [0, 0, 0]])
    """
    cleaned = label_image.copy()
    unique_labels = np.unique(cleaned)
    for label_id in unique_labels:
        if label_id == 0:  # Background
            continue
        mask = (cleaned == label_id)
        area = np.sum(mask)
        if area < min_area:
            cleaned[mask] = 0
    return cleaned


def aggregate_counts(predictions_folder: str, label_key: str = 'predictions') -> pd.DataFrame:
    """Aggregate segmentation statistics from a folder of predicted label images.

    Scans a folder for .npy files (assumed to be integer label images from instance
    segmentation) and computes per-image statistics: object count, mean area, std area.

    Args:
        predictions_folder: Path to folder containing .npy files with label images.
        label_key: Suffix/key to identify prediction files. Files like 'image_name_{label_key}.npy'
                   are processed. If not found, all .npy files are used.

    Returns:
        DataFrame with columns: ['image', 'count', 'mean_area', 'std_area']
        - image: filename (without .npy)
        - count: number of unique objects (excluding background label 0)
        - mean_area: mean object area in pixels
        - std_area: standard deviation of object area

    Example:
        >>> # Assuming predictions_folder contains:
        >>> # - sample_1_predictions.npy (labeled image)
        >>> # - sample_2_predictions.npy
        >>> df = aggregate_counts('predictions')
        >>> df
                 image  count  mean_area   std_area
        0  sample_1        12      145.3       67.8
        1  sample_2         8      203.4       91.2
    """
    results = []

    # Find all .npy files
    npy_files = sorted(Path(predictions_folder).glob('*.npy'))

    for npy_path in npy_files:
        filename = npy_path.stem  # Remove .npy

        try:
            label_image = np.load(str(npy_path))
        except Exception as e:
            print(f"Warning: Could not load {npy_path}: {e}")
            continue

        # Get unique labels (excluding 0 for background)
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            # No objects
            results.append({
                'image': filename,
                'count': 0,
                'mean_area': np.nan,
                'std_area': np.nan,
            })
            continue

        # Compute area for each object
        areas = []
        for label_id in unique_labels:
            area = np.sum(label_image == label_id)
            areas.append(area)

        areas = np.array(areas)
        results.append({
            'image': filename,
            'count': len(unique_labels),
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
        })

    df = pd.DataFrame(results)
    return df
