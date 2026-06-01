# Platelet Segmentation (platelet-seg)

A repository for instance segmentation of platelet alpha-granules from electron microscopy (EM) images. The project provides data tiling, training (StarDist or UNet fallback), inference utilities, an Optuna HPO driver, a Flask prediction API, and a small React viewer for inspecting results.

Goals
- Provide reproducible pipelines for training and evaluating instance segmentation models on EM platelet images.
- Offer tools for tiling large images, training/updating models, and running inference at scale.
- Make experimentation easy via an Optuna HPO driver and Docker development environment.

Repository Layout
```
src/
  active_learning.py        # active learning helpers
  augment.py                # augmentation pipelines
  dataset.py                # TileDataset and tiling utilities
  metrics.py                # IoU, PQ, F1 and evaluation utilities
  stardist_train.py         # training script (StarDist or UNet fallback)
  stardist_infer.py        # tiled inference utilities
  hpo_optuna.py             # Optuna HPO driver
  api/                     # Flask API (predict blueprint, app factory)
frontend/
  src/components/PredictionViewer.jsx  # React component for previewing predictions
docker/
  Dockerfile
  docker-compose.yml
tests/
  test_dataset.py
  test_metrics.py
  test_api_predict.py
.vscode/                    # recommended VS Code settings, tasks and launch configs
.pre-commit-config.yaml     # pre-commit hooks
README.md

```

Quickstart
----------
Prerequisites
- Python 3.10
- (Optional) NVIDIA GPU + drivers + NVIDIA Container Toolkit for GPU Docker runtime

1) Create a virtual environment and install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# If you have a requirements.txt, install it
pip install -r requirements.txt || echo "Install project deps manually."
# Essential testing/dev deps
pip install pytest numpy opencv-python
```

2) Dataset tiling (example)

The `src/dataset.py` file contains example code and CLI usage for generating tiles and a manifest. Adapt to your dataset manifest format.

```bash
python src/dataset.py --manifest data/manifest.csv --output_dir data/tiled --tile_size 512
```

3) Train (StarDist or UNet fallback)

Example training command (StarDist recommended if installed):

```bash
python src/stardist_train.py \
  --manifest data/manifest.csv \
  --checkpoint_dir experiments/stardist_model \
  --gpu 0 \
  --epochs 200 \
  --batch_size 4 \
  --lr 3e-4 \
  --patch_size 512 \
  --use_pretrained \
  --augment
```

UNet fallback (if StarDist is not installed) uses a `.pth` checkpoint workflow.

4) Inference (single image or folder)

```bash
python src/stardist_infer.py \
  --model experiments/stardist_model \
  --input data/images \
  --out_dir experiments/inference_results \
  --tile_size 1024 \
  --overlap 128 \
  --min_area_px 50 \
  --device cuda:0
```

5) Run the Flask API (development)

The Flask app factory is in `src/api/base.py`. To run locally:

```bash
# With virtualenv active
export FLASK_APP=src/api/base.py
flask run --host=0.0.0.0 --port=5000

# Or run the module directly
python src/api/base.py
```

The prediction endpoint is `POST /api/predict` and accepts a multipart file (`image`) or JSON `{ "image_path": "/abs/path/to/image.png" }`.

6) Frontend preview (React)

The repo includes a single-file React component `frontend/src/components/PredictionViewer.jsx` as an example viewer. To use it you will need a small React app (Create React App/Vite) and include the component. Example usage:

```jsx
import PredictionViewer from './components/PredictionViewer';

function App(){
  return <PredictionViewer API_BASE="http://localhost:5000" />;
}
```

VS Code & Docker Development Tips
---------------------------------
- VS Code
  - A `.vscode/` folder with `tasks.json`, `launch.json`, and `settings.json` is included. Use the `Run tests` task or debugger configurations for rapid iteration.
  - `python.defaultInterpreterPath` in `.vscode/settings.json` is set to `${workspaceFolder}/.venv/bin/python` (change for Windows to `.venv\\Scripts\\python.exe`).

- Docker
  - GPU image (recommended for training/inference on GPU hosts): `docker/Dockerfile` uses `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04`.
  - Build and run with docker-compose (GPU runtime):

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

  - CPU-only fallback: modify `docker/Dockerfile` to use `python:3.10-slim` (comment in file) and remove `runtime: nvidia` in `docker/docker-compose.yml`.

Hyperparameters & Tips (targeting >95% instance PQ)
----------------------------------------------------
Default hyperparameters (examples used in this repo):
- `patch_size`: 512
- `batch_size`: 4
- `learning_rate`: 3e-4 (fine-tune lower for pre-trained backbones, e.g., 1e-4 to 5e-5)
- `weight_decay`: 1e-5
- `epochs`: 200 (or early-stopped)
- `min_area_px`: 50 (post-filtering small noisy instances)

Checklist to reach high instance PQ (>95%)
- [ ] Use transfer learning (`--use_pretrained`) when training StarDist backbone.
- [ ] Heavy augmentation (rotation, intensity, scale) to improve robustness.
- [ ] Use a smaller learning rate when fine-tuning pretrained weights (1e-4 → 5e-5).
- [ ] Inspect failure modes: undersegmentation vs oversegmentation and tune `min_area_px` accordingly.
- [ ] Use Optuna HPO (`src/hpo_optuna.py`) to search lr, weight_decay, batch_size, and model width multipliers.
- [ ] Validate on a held-out patient/dataset split rather than random 80/20 splits.
- [ ] Increase training epochs with early-stopping on validation PQ.
- [ ] Consider ensembling or test-time augmentation for final deployment.

Running Tests & CI Notes
------------------------
Run tests locally with `pytest`:

```bash
# From repo root (with venv active)
pytest -q
```

CI recommendations
- Use a lightweight test matrix (unit tests only) on PRs to keep feedback fast.
- Run heavier integration tests (full training/inference) as separate workflows or guarded by labels.
- Example GitHub Actions steps:
  - Setup Python 3.10, install `pip install -r requirements.txt` + `pip install -r dev-requirements.txt`
  - Run `pre-commit run --all-files`
  - Run `pytest -q`

Large Models & Artifacts
------------------------
- Do not commit large model artifacts to Git. Use external storage:
  - Amazon S3, Google Cloud Storage, or Google Drive for large model checkpoints.
  - Use `git-lfs` if you need to keep model files in the repository but be mindful of storage costs.

Example: upload best checkpoint to S3 and reference a download script in `scripts/`:
```bash
aws s3 cp s3://my-bucket/models/best_model.pth models/checkpoints/best_model.pth
```

Contact & License
-----------------
- Contact: Your Name <your.email@example.com>
- License: Add a license file (e.g., `LICENSE`) and choose an open-source license (MIT/Apache-2.0). This README is a placeholder.

Acknowledgements
- Built on top of StarDist-style workflows and PyTorch/NumPy/OpenCV utilities.

If you want, I can also:
- Add a small CI workflow for GitHub Actions that runs tests and pre-commit.
- Add a `requirements.txt` / `dev-requirements.txt` with pinned dev dependencies.
- Add a short `scripts/download_model.sh` helper to fetch large models from S3/GDrive.

