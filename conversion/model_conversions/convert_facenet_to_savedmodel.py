"""
FaceNet model conversion utilities.

This module provides functions to convert FaceNet (InceptionResNetV1) models
to TensorFlow SavedModel format, with options to build from scratch and load
local .h5 weights or to use the project's load_facenet* functions that
download weights automatically.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import argparse
import os

# DeepFace deps
from deepface.commons import package_utils
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    import tensorflow as tf
    from keras.models import Model
else:
    import tensorflow as tf
    from tensorflow.keras.models import Model

# Import FaceNet builders from your project module
# Adjust the import path below if your module layout differs.
from deepface.models.facial_recognition.Facenet import (
    InceptionResNetV1,
    load_facenet128d_model,
    load_facenet512d_model,
)

# ---------------------------------------------------------------------

def build_facenet_model(dimension: int) -> Model:
    """
    Build FaceNet (InceptionResNetV1) model from scratch (no weights loaded).
    """
    return InceptionResNetV1(dimension=dimension)

def _select_loader(variant: str):
    """
    Pick the project's loader (downloads weights automatically).
    """
    v = variant.strip().lower()
    if v in ("128", "128d", "facenet128", "facenet-128d"):
        return load_facenet128d_model, 128
    if v in ("512", "512d", "facenet512", "facenet-512d"):
        return load_facenet512d_model, 512
    raise ValueError("Unknown variant. Use '128' or '512'.")

def _select_dimension(variant: str) -> int:
    _, dim = _select_loader(variant)
    return dim

def convert_facenet_to_savedmodel(
    export_dir: Union[str, Path],
    variant: str = "128",
    weights_path: Optional[Union[str, Path]] = None,
    use_load_model: bool = False,
    weights_url: Optional[str] = None,   # kept for API symmetry; project loaders already embed URLs
    verbose: bool = True,
) -> str:
    """
    Convert FaceNet (InceptionResNetV1) to SavedModel format.

    Args:
        export_dir: Directory to save the SavedModel.
        variant: '128' or '512' for embedding size.
        weights_path: Local .h5 weights path (only when use_load_model=False).
        use_load_model: If True, use project's load_facenet* (downloads weights).
                        If False, build from scratch and load local weights.
        weights_url: Optional custom URL (not typically needed; project loaders manage URLs).
        verbose: Verbose logging.

    Returns:
        str: Absolute path to the exported SavedModel directory.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Export directory: {export_dir.resolve()}")
        print(f"Variant: FaceNet-{_select_dimension(variant)}d")

    if use_load_model:
        loader, dim = _select_loader(variant)
        if verbose:
            print("Using project loader (will download weights if needed).")
            if weights_url:
                print("Note: weights_url parameter provided but project loaders already define URLs.")
        model = loader()
        if verbose:
            print("Model constructed and weights loaded via project loader.")
    else:
        dim = _select_dimension(variant)
        if weights_path is None:
            raise ValueError("weights_path must be provided when use_load_model=False")

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        if verbose:
            print(f"Building model (dimension={dim}) and loading weights from: {weights_path}")

        model = build_facenet_model(dimension=dim)
        # h5 weights expected to match layer names; enforce no mismatch to catch issues
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=False)
        if verbose:
            print("Weights loaded successfully.")

    # Warm up graph (FaceNet expects 160x160 RGB, channels-last)
    dummy = np.zeros((1, 160, 160, 3), dtype="float32")
    _ = model(dummy, training=False)

    if verbose:
        print("Model graph built.")

    # Keras 3 API (TF ≥ 2.14) has model.export; fallback for older TF/Keras
    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        model.save(str(export_dir), save_format="tf")

    if verbose:
        print(f"SavedModel written to: {export_dir.resolve()}")

    return str(export_dir.resolve())

# ---------------------------------------------------------------------

def main():
    """
    CLI for converting FaceNet to SavedModel.
    """
    parser = argparse.ArgumentParser(
        description="Convert FaceNet (InceptionResNetV1) model to SavedModel format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_root = os.getenv("FACENET_CONVERSION_ROOT", Path.cwd())
    default_export = os.getenv("FACENET_EXPORT_DIR", str(Path(default_root) / "facenet_saved_model"))

    parser.add_argument(
        "--variant", "-v",
        type=str,
        default=os.getenv("FACENET_VARIANT", "128"),
        choices=["128", "512"],
        help="FaceNet variant: 128 or 512 dimensional embeddings"
    )
    parser.add_argument(
        "--export-dir", "-o",
        type=str,
        default=default_export,
        help="Directory to save the SavedModel directory"
    )
    parser.add_argument(
        "--weights-path", "-w",
        type=str,
        default=os.getenv("FACENET_WEIGHTS_PATH", str(Path(default_root) / "facenet_weights.h5")),
        help="Path to local weights file (.h5). Only used if --use-local-weights is specified"
    )
    parser.add_argument(
        "--use-local-weights",
        action="store_true",
        help="Use local weights file instead of downloading via project loaders"
    )
    parser.add_argument(
        "--weights-url",
        type=str,
        help="Custom weights URL (not usually needed; kept for API parity)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    export_dir = Path(args.export_dir)

    if args.use_local_weights:
        wp = Path(args.weights_path)
        if not wp.exists():
            print(f"Warning: Weights file not found: {wp}")
            resp = input("Continue anyway? The function will raise an error. (y/N): ")
            if resp.lower() != "y":
                print("Aborted.")
                return
        else:
            print(f"Weights file found: {wp}")

    try:
        convert_facenet_to_savedmodel(
            export_dir=export_dir,
            variant=args.variant,
            weights_path=(Path(args.weights_path) if args.use_local_weights else None),
            use_load_model=not args.use_local_weights,
            weights_url=args.weights_url,
            verbose=not args.quiet
        )
        print("Conversion completed successfully!")
        print(f"SavedModel location: {export_dir}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()
