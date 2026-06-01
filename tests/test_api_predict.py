"""
Lightweight API tests that monkeypatch inference to avoid heavy model loads.

These tests patch `src.api.predict._run_inference` to return a simple
labels image and overlay so the Flask endpoints can be exercised quickly.
"""
import io
import json

import cv2
import numpy as np
import pytest


# Import module to patch before app creation
import src.api.predict as predict_module


# Dummy inference returns a small label image with two instances and an overlay
DUMMY_H = 128
DUMMY_W = 128


def _dummy_run_inference(image, model, model_type, device, tile_size, overlap, min_area_px):
    labels = np.zeros((DUMMY_H, DUMMY_W), dtype=np.uint16)
    cv2.circle(labels, (32, 32), 12, 1, -1)
    cv2.circle(labels, (96, 80), 10, 2, -1)

    overlay = np.zeros((DUMMY_H, DUMMY_W, 3), dtype=np.uint8)
    overlay[:] = 120
    cv2.circle(overlay, (32, 32), 12, (0, 255, 0), -1)
    cv2.circle(overlay, (96, 80), 10, (255, 0, 0), -1)

    return labels, overlay


@pytest.fixture
def app(monkeypatch):
    # Patch inference function so blueprint doesn't try to load a heavy model
    monkeypatch.setattr(predict_module, "_run_inference", _dummy_run_inference)

    # Import factory after patching
    from src.api.base import create_app

    app = create_app(model_path='dummy', device='cpu')
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_predict_multipart_file_upload(client):
    img = np.zeros((DUMMY_H, DUMMY_W), dtype=np.uint8)
    cv2.circle(img, (32, 32), 12, 255, -1)

    _, png = cv2.imencode('.png', img)
    data = {'image': (io.BytesIO(png.tobytes()), 'test.png')}

    resp = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'summary' in body and 'instances' in body


def test_predict_json_image_path(client, tmp_path):
    img = np.zeros((DUMMY_H, DUMMY_W), dtype=np.uint8)
    cv2.circle(img, (32, 32), 12, 255, -1)
    p = tmp_path / 'img.png'
    cv2.imwrite(str(p), img)

    resp = client.post('/api/predict', data=json.dumps({'image_path': str(p)}), content_type='application/json')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'summary' in body and 'instances' in body


def test_predict_missing_image(client):
    resp = client.post('/api/predict', data={})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['status'] == 'error'

