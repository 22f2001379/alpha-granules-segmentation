import argparse
from pathlib import Path

from src.dataset import build_manifest


def main():
    parser = argparse.ArgumentParser("Build tiled dataset")
    parser.add_argument("--input_images", required=True)
    parser.add_argument("--input_anns", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--save_empty", action="store_true")

    args = parser.parse_args()

    build_manifest(
        image_dir=Path(args.input_images),
        annotation_dir=Path(args.input_anns),
        out_dir=Path(args.out_dir),
        tile_size=args.tile_size,
        overlap=args.overlap,
        save_empty=args.save_empty,
    )


if __name__ == "__main__":
    main()
