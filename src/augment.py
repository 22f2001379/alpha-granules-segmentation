import os
from typing import Tuple

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise ImportError(
        "Albumentations is not installed. Please install it with 'pip install albumentations' to use this module."
    )


def get_train_transforms(tile_size: int = 512) -> A.Compose:
    """
    Get the augmentation pipeline for training EM grayscale tiles.

    These transforms are designed to be applied to both the image and its
    corresponding segmentation mask. The mask is handled correctly using
    nearest-neighbor interpolation to preserve integer labels.

    Args:
        tile_size: The target size of the output tile (height and width).

    Returns:
        An Albumentations composition of transforms.
    """
    return A.Compose(
        [
            # Geometric transforms
            A.RandomResizedCrop(
                height=tile_size,
                width=tile_size,
                scale=(0.9, 1.1),
                ratio=(0.9, 1.1),
                p=0.9,
            ),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.7,
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),

            # Pixel-level transforms
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.5,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.75
            ),

            # Final conversion to tensor
            # Normalizes image to [0, 1] and converts both image and mask to CxHxW format.
            # Mask remains as integer-like floats (0.0, 1.0).
            ToTensorV2(),
        ]
    )


def get_val_transforms(tile_size: int = 512) -> A.Compose:
    """
    Get the validation augmentation pipeline for EM grayscale tiles.

    This pipeline center-crops the image and mask to the target tile size
    and converts them to PyTorch tensors.

    Args:
        tile_size: The target size of the output tile (height and width).

    Returns:
        An Albumentations composition of transforms.
    """
    return A.Compose(
        [
            A.CenterCrop(height=tile_size, width=tile_size, p=1.0),
            ToTensorV2(),
        ]
    )


if __name__ == "__main__":
    # This block serves as a visual test for the training augmentations.
    # It loads a sample image and mask, applies the transforms multiple times,
    # and saves a grid of the results to 'experiments/aug_test.png'.

    # Ensure torch and torchvision are installed for this test block
    try:
        import torch
        import torchvision
        from torchvision.utils import make_grid
    except ImportError:
        print(
            "Torch or torchvision not found. Please run 'pip install torch torchvision' to run the test block."
        )
        exit()

    # --- Configuration ---
    SAMPLE_IMG_PATH = "data/manifests/tiles/images/sample.png"
    SAMPLE_MASK_PATH = "data/manifests/tiles/masks/sample.png"
    OUTPUT_PATH = "experiments/aug_test.png"
    TILE_SIZE = 512
    NUM_EXAMPLES = 6

    # --- Loading Data ---
    print(f"Loading sample image from: {SAMPLE_IMG_PATH}")
    print(f"Loading sample mask from: {SAMPLE_MASK_PATH}")

    if not os.path.exists(SAMPLE_IMG_PATH) or not os.path.exists(SAMPLE_MASK_PATH):
        print(
            "Error: Sample image or mask not found. Please ensure the following files exist:"
        )
        print(f"- {SAMPLE_IMG_PATH}")
        print(f"- {SAMPLE_MASK_PATH}")
        # Create dummy files for demonstration if they don't exist
        print("Creating dummy image and mask for demonstration purposes.")
        os.makedirs(os.path.dirname(SAMPLE_IMG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(SAMPLE_MASK_PATH), exist_ok=True)
        dummy_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        dummy_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        cv2.circle(dummy_mask, (TILE_SIZE // 2, TILE_SIZE // 2), 100, 1, -1) # A circle as a mask
        cv2.imwrite(SAMPLE_IMG_PATH, dummy_tile)
        cv2.imwrite(SAMPLE_MASK_PATH, dummy_mask)
        print("Dummy files created.")


    # Albumentations expects images in Numpy format (H, W, C) and BGR channel order.
    # cv2.imread loads images in BGR format by default.
    # For grayscale, we load as-is and it will be (H, W). Albumentations handles it.
    image = cv2.imread(SAMPLE_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(SAMPLE_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
         raise IOError(f"Could not load image or mask from the specified paths. Check the files.")

    # Ensure mask is binary (0 or 255) and convert to (0, 1)
    mask = (mask > 128).astype(np.uint8)

    # --- Applying Transforms ---
    print(f"Generating {NUM_EXAMPLES} augmented examples...")
    train_transforms = get_train_transforms(tile_size=TILE_SIZE)
    augmented_examples = []

    for _ in range(NUM_EXAMPLES):
        augmented = train_transforms(image=image, mask=mask)
        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        # For visualization, combine image and mask.
        # We'll stack the grayscale image into 3 channels to make it RGB,
        # then overlay the mask in a color (e.g., red).
        img_rgb = aug_img.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
        mask_overlay = torch.zeros_like(img_rgb)
        mask_overlay[0, :, :] = aug_mask  # Red channel for mask
        
        # Create a blended visualization
        # We'll show the image with a 50% transparent red overlay for the mask
        blended = img_rgb * 0.7 + mask_overlay * 0.3
        augmented_examples.append(blended)

    # --- Saving Grid ---
    print(f"Saving augmentation grid to: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # make_grid expects a list of tensors
    grid = make_grid(augmented_examples, nrow=3, padding=10, pad_value=0.5)
    
    # save_image handles the conversion from tensor to image file
    torchvision.utils.save_image(grid, OUTPUT_PATH)

    print("Done. Check the 'experiments' folder for the output image.")
