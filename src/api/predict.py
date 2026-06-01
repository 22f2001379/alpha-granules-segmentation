"""
Prediction API endpoint for platelet-seg inference.

Provides a Flask blueprint with POST /predict endpoint that accepts images
and returns instance segmentation results in JSON format.
"""
import base64
import io
import os
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Blueprint, jsonify, request

# Import inference utilities
from src.stardist_infer import load_model, calculate_instance_stats, filter_small_instances

# Lock for thread-safe model inference
_inference_lock = threading.Lock()


def create_predict_blueprint(model_path: str, device: str = 'cpu') -> Blueprint:
    """
    Create and return a Flask blueprint for prediction endpoints.

    The model is loaded once at blueprint creation and reused for all requests
    (thread-safe via a lock around inference).

    Args:
        model_path: Path to trained model (directory for StarDist or .pth file for UNet).
        device: Device string, e.g., 'cpu', 'cuda:0', 'cuda:1'.

    Returns:
        Flask Blueprint with POST /predict endpoint.
    """
    blueprint = Blueprint('predict', __name__)

    # Load model once
    device_obj = torch.device(device)
    try:
        model, model_type = load_model(model_path, device_obj)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    print(f"Loaded {model_type} model on device {device}")

    @blueprint.route('/predict', methods=['POST'])
    def predict():
        """
        Perform instance segmentation inference.

        Accepts:
        - multipart/form-data with 'image' file, or
        - JSON with 'image_path' string

        Optional parameters (query string or form):
        - tile_size (int): Tile size for large images (default: 1024)
        - overlap (int): Overlap between tiles in pixels (default: 128)
        - min_area_px (int): Minimum object area to keep (default: 50)
        - overlay_base64 (bool): Include base64-encoded overlay PNG (default: false)

        Returns JSON:
        {
            "status": "success",
            "summary": {
                "count": <int>,
                "mean_area": <float>,
                "std_area": <float>
            },
            "instances": [
                {
                    "id": <int>,
                    "centroid": [<x>, <y>],
                    "area_px": <int>,
                    "bbox": [<x1>, <y1>, <x2>, <y2>]
                },
                ...
            ],
            "overlay_base64": "data:image/png;base64,..." // optional
        }
        """
        try:
            # Parse parameters
            tile_size = request.args.get('tile_size', 1024, type=int)
            overlap = request.args.get('overlap', 128, type=int)
            min_area_px = request.args.get('min_area_px', 50, type=int)
            include_overlay = request.args.get('overlay_base64', 'false').lower() == 'true'

            # Validate tile parameters
            if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid tile_size or overlap parameters.'
                }), 400

            # Load image
            image_array = None
            if 'image' in request.files:
                # From multipart file
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'status': 'error', 'message': 'No file provided.'}), 400
                try:
                    file_bytes = file.read()
                    image_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image_array is None:
                        raise ValueError("Could not decode image.")
                except Exception as e:
                    return jsonify({'status': 'error', 'message': f'Image decode failed: {e}'}), 400

            elif request.is_json and 'image_path' in request.json:
                # From file path in JSON
                image_path = request.json['image_path']
                if not os.path.exists(image_path):
                    return jsonify({'status': 'error', 'message': f'Image file not found: {image_path}'}), 404
                try:
                    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image_array is None:
                        raise ValueError("Could not read image.")
                except Exception as e:
                    return jsonify({'status': 'error', 'message': f'Image read failed: {e}'}), 400
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Provide either multipart "image" file or JSON {"image_path": "..."}.'
                }), 400

            # Validate image dimensions
            max_dim = 16384  # Sanity limit
            if image_array.shape[0] > max_dim or image_array.shape[1] > max_dim:
                return jsonify({
                    'status': 'error',
                    'message': f'Image dimensions exceed maximum ({max_dim}x{max_dim}).'
                }), 413

            # Thread-safe inference
            with _inference_lock:
                labels, overlay_image = _run_inference(
                    image_array, model, model_type, device_obj, tile_size, overlap, min_area_px
                )

            # Compute statistics
            stats_df = calculate_instance_stats(labels, image_array)

            # Build response
            response_data = {
                'status': 'success',
                'summary': {
                    'count': int(labels.max()),
                    'mean_area': float(stats_df['area_px'].mean()) if len(stats_df) > 0 else 0.0,
                    'std_area': float(stats_df['area_px'].std()) if len(stats_df) > 0 else 0.0,
                },
                'instances': [],
            }

            # Populate instances
            if len(stats_df) > 0:
                for _, row in stats_df.iterrows():
                    response_data['instances'].append({
                        'id': int(row['instance_id']),
                        'centroid': [float(row['centroid_x']), float(row['centroid_y'])],
                        'area_px': int(row['area_px']),
                        'bbox': [
                            int(row['bbox_x1']), int(row['bbox_y1']),
                            int(row['bbox_x2']), int(row['bbox_y2'])
                        ],
                    })

            # Optional overlay
            if include_overlay and overlay_image is not None:
                _, png_data = cv2.imencode('.png', overlay_image)
                overlay_b64 = base64.b64encode(png_data).decode('utf-8')
                response_data['overlay_base64'] = f'data:image/png;base64,{overlay_b64}'

            return jsonify(response_data), 200

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    return blueprint


def _run_inference(
    image: np.ndarray,
    model: Any,
    model_type: str,
    device: torch.device,
    tile_size: int,
    overlap: int,
    min_area_px: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run inference on a single image using the provided model.

    Returns:
        (labels_image, overlay_image) where overlay is optional for visualization.
    """
    if model_type == 'stardist':
        # Use StarDist's built-in tiled prediction
        try:
            labels, _ = model.predict_instances_big(
                image,
                axes='YX',
                block_size=tile_size,
                min_overlap=overlap,
                context=overlap // 2,
                show_progress=False,
            )
        except Exception as e:
            raise RuntimeError(f"StarDist inference failed: {e}")
    else:
        # UNet manual tiling
        labels = _unet_tiled_inference(image, model, device, tile_size, overlap)

    # Post-processing
    if min_area_px > 0:
        labels = filter_small_instances(labels, min_area_px)

    # Generate overlay for visualization
    overlay = _create_overlay(image, labels)

    return labels, overlay


def _unet_tiled_inference(
    image: np.ndarray,
    model: nn.Module,
    device: torch.device,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """Run tiled inference for UNet model and return label image."""
    from scipy.ndimage import label as scipy_label
    from skimage.segmentation import watershed

    h, w = image.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

    stitched_heatmap = np.zeros(padded_image.shape, dtype=np.float32)
    norm_mask = np.zeros(padded_image.shape, dtype=np.float32)

    y_steps = range(0, padded_image.shape[0], tile_size - overlap)
    x_steps = range(0, padded_image.shape[1], tile_size - overlap)

    for y in y_steps:
        for x in x_steps:
            if y + tile_size > padded_image.shape[0] or x + tile_size > padded_image.shape[1]:
                continue
            tile = padded_image[y : y + tile_size, x : x + tile_size]
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                output = model(tile_tensor)

            heatmap = torch.sigmoid(output).squeeze().cpu().numpy()
            stitched_heatmap[y : y + tile_size, x : x + tile_size] += heatmap
            norm_mask[y : y + tile_size, x : x + tile_size] += 1

    stitched_heatmap /= (norm_mask + 1e-6)
    stitched_heatmap = stitched_heatmap[:h, :w]

    thresholded = stitched_heatmap > 0.5
    markers, _ = scipy_label(thresholded)
    labels = watershed(-stitched_heatmap, markers, mask=thresholded)

    return labels


def _create_overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Create an RGB overlay visualization of labels on top of image."""
    try:
        from stardist.plot import render_label

        # Normalize image to 8-bit
        if image.dtype != np.uint8:
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-6)
            img_8bit = (img_norm * 255).astype(np.uint8)
        else:
            img_8bit = image

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)

        # Render overlay
        overlay = render_label(labels, img=img_rgb, alpha=0.5)
        return overlay
    except Exception as e:
        print(f"Warning: Could not create overlay: {e}")
        return None
