import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================
# Debugging tool for mask preprocessing
# =============================
# This script loads a single EM annotation and its matching grayscale image,
# visualizes each preprocessing stage, and saves all intermediates for inspection.
#
# Motivation: StarDist requires FILLED instance masks, but current pipeline
# produces hollow contours due to imperfect annotation outlines and unreliable flood fill.
# This script replaces flood fill with robust contour-based filling and provides
# visual diagnostics for each step.

# -----------------------------
# CONFIGURATION
# -----------------------------
DEBUG_OUTDIR = Path("experiments/debug_masks")
DEBUG_OUTDIR.mkdir(parents=True, exist_ok=True)

# Tune these paths for your test case
RAW_ANN_PATH = Path("data/annotations_debug/24-388b_16.jpg")  # <-- set to a real annotation
RAW_IMG_PATH = Path("data/raw_debug/24-388b_16.jpg")       # <-- set to a real EM image

# HSV color thresholds for blue (granule) and yellow (OCS exclusion)
BLUE_HSV = dict(hue=(85, 150), sat=(30, 255), val=(30, 255))
YELLOW_HSV = dict(hue=(20, 45), sat=(30, 255), val=(30, 255))

# Morphology
MORPH_KERNEL_SIZE = 5  # elliptical kernel size for closing
MIN_AREA = 30          # minimum area for valid granule
MAX_AREA = 2000        # maximum area for valid granule
MAX_ASPECT = 4.0       # max aspect ratio for valid granule

# -----------------------------
# Utility functions
# -----------------------------
def save_img(name, arr):
    out_path = DEBUG_OUTDIR / f"{name}.png"
    cv2.imwrite(str(out_path), arr if arr.dtype == np.uint8 else (arr * 255).astype(np.uint8))
    print(f"[DEBUG] Saved: {out_path}")

# -----------------------------
# Load images
# -----------------------------
assert RAW_ANN_PATH.exists(), f"Annotation not found: {RAW_ANN_PATH}"
assert RAW_IMG_PATH.exists(), f"Image not found: {RAW_IMG_PATH}"

ann = cv2.imread(str(RAW_ANN_PATH))
img = cv2.imread(str(RAW_IMG_PATH), cv2.IMREAD_GRAYSCALE)
save_img("00_annotation_rgb", ann)
save_img("00_em_image", img)

# -----------------------------
# HSV color extraction
# -----------------------------
hsv = cv2.cvtColor(ann, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Blue mask (alpha granule outlines)
blue_mask = (
    (h >= BLUE_HSV['hue'][0]) & (h <= BLUE_HSV['hue'][1]) &
    (s >= BLUE_HSV['sat'][0]) & (s <= BLUE_HSV['sat'][1]) &
    (v >= BLUE_HSV['val'][0]) & (v <= BLUE_HSV['val'][1])
).astype(np.uint8)
save_img("01_blue_mask", blue_mask * 255)

# Yellow mask (OCS exclusion)
yellow_mask = (
    (h >= YELLOW_HSV['hue'][0]) & (h <= YELLOW_HSV['hue'][1]) &
    (s >= YELLOW_HSV['sat'][0]) & (s <= YELLOW_HSV['sat'][1]) &
    (v >= YELLOW_HSV['val'][0]) & (v <= YELLOW_HSV['val'][1])
).astype(np.uint8)
save_img("02_yellow_mask", yellow_mask * 255)

# -----------------------------
# Clean blue mask (remove speckles, close gaps)
# -----------------------------
# Remove yellow from blue (exclusion)
cleaned = blue_mask.copy()
cleaned[yellow_mask > 0] = 0
save_img("03_cleaned_blue_exclude_yellow", cleaned * 255)

# Morphological closing (elliptical kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
save_img("04_morph_closed", closed * 255)

# -----------------------------
# Contour detection and visualization
# -----------------------------
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_vis = ann.copy()
cv2.drawContours(contour_vis, contours, -1, (0, 0, 255), 2)
save_img("05_contours", contour_vis)
print(f"[DEBUG] Found {len(contours)} contours.")

# -----------------------------
# Contour-based filling
# -----------------------------
filled_mask = np.zeros_like(closed)
areas = []
retained = 0
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = max(w / (h + 1e-6), h / (w + 1e-6))
    if aspect > MAX_ASPECT:
        continue
    cv2.drawContours(filled_mask, [cnt], -1, 1, thickness=cv2.FILLED)
    areas.append(area)
    retained += 1

save_img("06_filled_mask", filled_mask * 255)
print(f"[DEBUG] Retained {retained} biologically plausible contours.")
if areas:
    print(f"[DEBUG] Area stats: min={np.min(areas):.1f}, max={np.max(areas):.1f}, mean={np.mean(areas):.1f}")
else:
    print("[DEBUG] No valid contours after filtering.")

# -----------------------------
# Final binary mask
# -----------------------------
final_mask = filled_mask.copy()
save_img("07_final_binary_mask", final_mask * 255)

# -----------------------------
# Side-by-side matplotlib debug panel
# -----------------------------
fig, axs = plt.subplots(2, 4, figsize=(18, 9))
axs = axs.ravel()

axs[0].imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Annotation")
axs[1].imshow(blue_mask, cmap="Blues")
axs[1].set_title("Blue Mask")
axs[2].imshow(yellow_mask, cmap="Wistia")
axs[2].set_title("Yellow Mask")
axs[3].imshow(cleaned, cmap="gray")
axs[3].set_title("Cleaned (Blue - Yellow)")
axs[4].imshow(closed, cmap="gray")
axs[4].set_title("Morph Closed")
axs[5].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
axs[5].set_title("Contours")
axs[6].imshow(filled_mask, cmap="gray")
axs[6].set_title("Filled Mask")
axs[7].imshow(final_mask, cmap="gray")
axs[7].set_title("Final Binary Mask")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
panel_path = DEBUG_OUTDIR / "08_debug_panel.png"
plt.savefig(panel_path, dpi=200)
print(f"[DEBUG] Saved debug panel: {panel_path}")

# -----------------------------
# Commentary
# -----------------------------
# Why each step exists:
# - HSV extraction: robust to annotation color variations, avoids RGB artifacts.
# - Blue/yellow masks: separate target (granule) from exclusion (OCS) regions.
# - Morph closing: bridges small gaps in hand-drawn contours, but elliptical kernel avoids over-dilation.
# - Contour detection: finds all closed shapes, even if imperfect.
# - Contour-based filling: fills each detected region, robust to open contours if gaps are small.
# - Biological filtering: removes implausible objects (tiny, huge, elongated).
# - Flood fill is avoided because open contours cause leaks, producing hollow or incomplete masks.
# - Saving intermediates: enables visual debugging of every stage.

# Why contour filling is preferred:
# - More robust to imperfect hand-drawn outlines.
# - Each region is filled independently, so partial gaps do not destroy the mask.
# - Easier to debug and tune visually.

# Parameters to tune visually:
# - HSV thresholds for blue/yellow
# - Morph kernel size and iterations
# - Area/aspect ratio thresholds

# To run: adjust RAW_ANN_PATH and RAW_IMG_PATH to point to a real annotation/image pair.
# All outputs will be saved to experiments/debug_masks/.

# End of script.
