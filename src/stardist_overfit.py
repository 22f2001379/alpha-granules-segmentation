import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stardist.models import StarDist2D, Config2D
from csbdeep.utils import normalize


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/manifests/debug")
MODEL_DIR = Path("experiments/stardist_overfit")
MODEL_NAME = "alpha_granules_overfit"

N_SAMPLES = 20
EPOCHS = 10
BATCH_SIZE = 2


# ============================================================
# LOAD DATA
# ============================================================

image_paths = sorted((DATA_DIR / "images").glob("*.png"))[:N_SAMPLES]
mask_paths  = sorted((DATA_DIR / "masks").glob("*.png"))[:N_SAMPLES]

assert len(image_paths) == len(mask_paths) > 0, "No data found"

X = [normalize(plt.imread(str(p))) for p in image_paths]
Y = [(plt.imread(str(p)) > 0).astype(np.uint16) for p in mask_paths]


# ============================================================
# STAR DIST CONFIG
# ============================================================

config = Config2D(
    n_rays=32,
    grid=(2, 2),
    train_batch_size=BATCH_SIZE,
    use_gpu=False,
)


# ============================================================
# MODEL
# ============================================================

model = StarDist2D(
    config,
    name=MODEL_NAME,
    basedir=str(MODEL_DIR),
)

model.train(
    X,
    Y,
    validation_data=(X, Y),
    epochs=EPOCHS,
)




# ============================================================
# VISUAL SANITY CHECK
# ============================================================

# pick a tile that actually contains alpha granules
idx = next(i for i, m in enumerate(Y) if m.sum() > 0)

test_img = X[idx]
test_mask = Y[idx]

labels, _ = model.predict_instances(test_img)


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(test_img, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("GT Mask")
plt.imshow(test_mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(labels, cmap="nipy_spectral")
plt.axis("off")

plt.tight_layout()
out_path = MODEL_DIR / MODEL_NAME / "overfit_prediction.png"
plt.savefig(out_path, dpi=200)
print(f"Saved prediction visualization to: {out_path}")

