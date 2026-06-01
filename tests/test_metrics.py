import numpy as np

from src.metrics import iou_matrix, compute_pq, compute_f1


def test_iou_and_pq_perfect_match():
    # Simple perfect match: one GT instance equals one predicted instance
    gt = np.zeros((32, 32), dtype=np.uint16)
    pred = np.zeros((32, 32), dtype=np.uint16)

    gt[5:15, 5:15] = 1
    pred[5:15, 5:15] = 1

    iou = iou_matrix(gt, pred)
    assert iou.shape == (1, 1)
    assert np.isclose(iou[0, 0], 1.0)

    pq = compute_pq(gt, pred, iou_thresh=0.5)
    assert np.isclose(pq['pq'], 1.0)
    assert np.isclose(pq['sq'], 1.0)
    assert np.isclose(pq['rq'], 1.0)

    f1 = compute_f1(gt, pred, iou_thresh=0.5)
    assert np.isclose(f1, 1.0)


def test_iou_and_pq_partial_overlap():
    # Partial overlap that is below threshold should produce FP and FN
    gt = np.zeros((40, 40), dtype=np.uint16)
    pred = np.zeros((40, 40), dtype=np.uint16)

    gt[5:18, 5:18] = 1
    pred[20:33, 20:33] = 1  # far apart

    iou = iou_matrix(gt, pred)
    # IoU should be zero (no overlap)
    assert iou.size == 1
    assert np.isclose(iou[0, 0], 0.0)

    pq = compute_pq(gt, pred, iou_thresh=0.5)
    # No true positives -> pq, sq, rq should be 0
    assert np.isclose(pq['pq'], 0.0)
    assert np.isclose(pq['sq'], 0.0)
    assert np.isclose(pq['rq'], 0.0)
