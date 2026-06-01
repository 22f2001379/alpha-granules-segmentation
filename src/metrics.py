"""
Module for computing instance-level segmentation metrics.

This module provides functions to calculate common instance segmentation metrics
such as Panoptic Quality (PQ), F1-score, and Precision/Recall at various
Intersection over Union (IoU) thresholds.

It uses the Hungarian algorithm for optimal instance matching and is designed
to handle edge cases like the absence of ground truth or predicted labels.
A dataset-level aggregation function is also provided to summarize results
over multiple images.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Computes the Intersection over Union (IoU) matrix between ground truth and predicted instances.

    Args:
        gt_labels: A 2D NumPy array of shape (H, W) with integer-labeled ground truth instances.
        pred_labels: A 2D NumPy array of shape (H, W) with integer-labeled predicted instances.

    Returns:
        A NumPy array of shape (M, N) where M is the number of unique ground truth instances
        and N is the number of unique predicted instances. The entry (i, j) is the IoU
        between ground truth instance i+1 and predicted instance j+1.
    """
    gt_ids = np.unique(gt_labels[gt_labels > 0])
    pred_ids = np.unique(pred_labels[pred_labels > 0])

    if gt_ids.size == 0 or pred_ids.size == 0:
        return np.array([[]])

    iou_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    for i, gt_id in enumerate(gt_ids):
        gt_mask = (gt_labels == gt_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_labels == pred_id)
            
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            
            if union > 0:
                iou_mat[i, j] = intersection / union

    return iou_mat


def _get_matching_stats(
    gt_labels: np.ndarray, pred_labels: np.ndarray, iou_thresh: float
) -> Tuple[int, int, int, List[Tuple[int, int]], List[float]]:
    """
    Helper function to compute matching statistics (TP, FP, FN) for a given IoU threshold.

    Returns:
        A tuple containing:
        - tp (int): Number of True Positives.
        - fp (int): Number of False Positives.
        - fn (int): Number of False Negatives.
        - matched_pairs (list): A list of (gt_id, pred_id) tuples for matched instances.
        - iou_of_matches (list): A list of IoU values for each matched pair.
    """
    gt_ids = np.unique(gt_labels[gt_labels > 0])
    pred_ids = np.unique(pred_labels[pred_labels > 0])
    num_gt = len(gt_ids)
    num_pred = len(pred_ids)

    if num_gt == 0 and num_pred == 0:
        return 0, 0, 0, [], []
    if num_gt == 0:
        return 0, num_pred, 0, [], []
    if num_pred == 0:
        return 0, 0, num_gt, [], []

    iou_mat = iou_matrix(gt_labels, pred_labels)
    
    # Use Hungarian algorithm to find the optimal assignment
    # We want to maximize IoU, so we use the negative IoU as cost
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    # Filter matches based on the IoU threshold
    matched_iou = iou_mat[row_ind, col_ind]
    is_match = matched_iou >= iou_thresh
    
    tp = np.sum(is_match)
    fp = num_pred - tp
    fn = num_gt - tp

    matched_pairs = []
    iou_of_matches = []
    if tp > 0:
        matched_gt_indices = row_ind[is_match]
        matched_pred_indices = col_ind[is_match]
        matched_pairs = list(zip(gt_ids[matched_gt_indices], pred_ids[matched_pred_indices]))
        iou_of_matches = list(matched_iou[is_match])

    return int(tp), int(fp), int(fn), matched_pairs, iou_of_matches


def compute_f1(gt_labels: np.ndarray, pred_labels: np.ndarray, iou_thresh: float = 0.5) -> float:
    """
    Computes the F1-score for instance segmentation.

    F1 = 2 * TP / (2 * TP + FP + FN)
    """
    tp, fp, fn, _, _ = _get_matching_stats(gt_labels, pred_labels, iou_thresh)
    denominator = (2 * tp + fp + fn)
    return (2 * tp) / denominator if denominator > 0 else 0.0


def compute_pq(
    gt_labels: np.ndarray, pred_labels: np.ndarray, iou_thresh: float = 0.5
) -> Dict[str, float]:
    """
    Computes Panoptic Quality (PQ) and its components, Segmentation Quality (SQ) and Recognition Quality (RQ).
    """
    tp, fp, fn, _, iou_of_matches = _get_matching_stats(gt_labels, pred_labels, iou_thresh)
    
    # Segmentation Quality (SQ)
    sq = sum(iou_of_matches) / tp if tp > 0 else 0.0
    
    # Recognition Quality (RQ)
    rq_denominator = tp + 0.5 * fp + 0.5 * fn
    rq = tp / rq_denominator if rq_denominator > 0 else 0.0
    
    # Panoptic Quality (PQ)
    pq = sq * rq
    
    return {'pq': pq, 'sq': sq, 'rq': rq}


def compute_stats_at_thresholds(
    gt_labels: np.ndarray, pred_labels: np.ndarray, thresholds: List[float]
) -> Dict[float, Dict[str, float]]:
    """
    Computes F1, Precision, and Recall at multiple IoU thresholds.

    Note: This is not the COCO-style Average Precision (AP), which requires
    confidence scores for each prediction and integration over a recall range.
    This function provides a snapshot of performance at fixed IoU thresholds.
    """
    results = {}
    for thresh in thresholds:
        tp, fp, fn, _, _ = _get_matching_stats(gt_labels, pred_labels, thresh)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[thresh] = {'f1': f1, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn}
    return results


def dataset_aggregate(metrics_list: List[Dict[str, Any]], out_dir: str):
    """
    Aggregates metrics over a dataset and saves summary reports.

    Args:
        metrics_list: A list where each item is a dictionary of metrics for one image.
                      Each dictionary should include a 'filename' key.
        out_dir: The directory where the report files will be saved.
    """
    if not metrics_list:
        print("Warning: metrics_list is empty. No report will be generated.")
        return

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(metrics_list)
    
    # Save detailed per-image JSON report
    json_path = out_path / "per_image_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    print(f"Saved detailed report to: {json_path}")

    # Create and save summary CSV
    summary_df = df.describe().transpose()[['mean', 'std']]
    csv_path = out_path / "summary_metrics.csv"
    summary_df.to_csv(csv_path)
    print(f"Saved summary report to: {csv_path}")


if __name__ == "__main__":
    print("--- Running Metrics Module Demonstration ---")

    # 1. Create synthetic ground truth and prediction images
    H, W = 300, 300
    gt = np.zeros((H, W), dtype=np.uint16)
    pred = np.zeros((H, W), dtype=np.uint16)

    # GT instances
    cv2.circle(gt, (50, 50), 30, 1, -1)   # GT 1: Matched (TP)
    cv2.circle(gt, (150, 150), 40, 2, -1) # GT 2: Missed (FN)
    cv2.circle(gt, (230, 80), 25, 3, -1)  # GT 3: Poorly matched (FN at high IoU)

    # Prediction instances
    cv2.circle(pred, (55, 55), 28, 1, -1)  # Pred 1: Good match for GT 1 (TP)
    cv2.circle(pred, (250, 250), 35, 2, -1) # Pred 2: Spurious detection (FP)
    cv2.circle(pred, (235, 85), 15, 3, -1) # Pred 3: Poor match for GT 3

    # 2. Compute and display the IoU matrix
    iou_mat = iou_matrix(gt, pred)
    print("\n1. IoU Matrix (GT instances x Predicted instances):")
    print(pd.DataFrame(iou_mat, index=[f"GT_{i}" for i in [1,2,3]], columns=[f"Pred_{i}" for i in [1,2,3]]))
    
    # 3. Compute metrics at a single threshold (0.5)
    iou_threshold = 0.5
    print(f"\n2. Metrics at IoU Threshold = {iou_threshold}:")
    
    f1_score = compute_f1(gt, pred, iou_thresh=iou_threshold)
    print(f"   - F1-Score: {f1_score:.4f}")

    pq_metrics = compute_pq(gt, pred, iou_thresh=iou_threshold)
    print(f"   - Panoptic Quality (PQ): {pq_metrics['pq']:.4f}")
    print(f"   - Segmentation Quality (SQ): {pq_metrics['sq']:.4f}")
    print(f"   - Recognition Quality (RQ): {pq_metrics['rq']:.4f}")

    # 4. Compute stats at multiple thresholds
    thresholds = [0.3, 0.5, 0.75]
    print(f"\n3. Stats at Multiple Thresholds {thresholds}:")
    ap_stats = compute_stats_at_thresholds(gt, pred, thresholds=thresholds)
    for thresh, stats in ap_stats.items():
        print(f"   - IoU Thresh={thresh}: Precision={stats['precision']:.2f}, Recall={stats['recall']:.2f}, F1={stats['f1']:.2f} "
              f"(TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']})")

    # 5. Demonstrate dataset aggregation
    print("\n4. Dataset Aggregation Demonstration:")
    
    # Create a list of dummy metrics for two images
    all_metrics = [
        {
            'filename': 'image_01.png',
            'f1_at_0.5': f1_score,
            'pq': pq_metrics['pq']
        },
        {
            'filename': 'image_02.png',
            'f1_at_0.5': 0.85, # Dummy value
            'pq': 0.75,       # Dummy value
        }
    ]
    
    output_directory = "experiments/metrics_report"
    dataset_aggregate(all_metrics, out_dir=output_directory)
    print("\n--- Demonstration Complete ---")
