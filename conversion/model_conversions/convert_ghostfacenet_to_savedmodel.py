"""
GhostFaceNet model conversion utility.

This script converts GhostFaceNetV1 models to TensorFlow SavedModel format,
with options to build from scratch and load local .h5 weights or to use the
default loader (downloads automatically).
"""

from pathlib import Path
from typing import Union, Optional
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

# Import GhostFaceNet builders
from deepface.models.facial_recognition.GhostFaceNet import (
    GhostFaceNetV1,
    load_model as load_ghostfacenet_model,
)

# ---------------------------------------------------------------------

def build_ghostfacenet_model() -> Model:
    """
    Build GhostFaceNetV1 model from scratch (no weights loaded).
    """
    return GhostFaceNetV1()

def convert_ghostfacenet_to_savedmodel(
    export_dir: Union[str, Path],
    weights_path: Optional[Union[str, Path]] = None,
    use_load_model: bool = False,
    verbose: bool = True,
) -> str:
    """
    Convert GhostFaceNetV1 to SavedModel format.

    Args:
        export_dir: Directory to save the SavedModel.
        weights_path: Local .h5 weights path (only when use_load_model=False).
        use_load_model: If True, use project loader (downloads weights).
                        If False, build from scratch and load local weights.
        verbose: Verbose logging.

    Returns:
        str: Absolute path to the exported SavedModel directory.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Export directory: {export_dir.resolve()}")
        print("Model: GhostFaceNetV1 (embedding=512d)")

    if use_load_model:
        if verbose:
            print("Using project loader (will download weights if needed).")
        model = load_ghostfacenet_model()
        if verbose:
            print("Model constructed and weights loaded via project loader.")
    else:
        if weights_path is None:
            raise ValueError("weights_path must be provided when use_load_model=False")

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        if verbose:
            print(f"Building model and loading weights from: {weights_path}")

        model = build_ghostfacenet_model()
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=False)
        if verbose:
            print("Weights loaded successfully.")

    # Warm up graph (GhostFaceNet expects 112x112 RGB)
    dummy = np.zeros((1, 112, 112, 3), dtype="float32")
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
    CLI for converting GhostFaceNet to SavedModel.
    """
    parser = argparse.ArgumentParser(
        description="Convert GhostFaceNetV1 model to SavedModel format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_root = os.getenv("GHOSTFACENET_CONVERSION_ROOT", Path.cwd())
    default_export = os.getenv("GHOSTFACENET_EXPORT_DIR", str(Path(default_root) / "ghostfacenet_saved_model"))

    parser.add_argument(
        "--export-dir", "-o",
        type=str,
        default=default_export,
        help="Directory to save the SavedModel directory"
    )
    parser.add_argument(
        "--weights-path", "-w",
        type=str,
        default=os.getenv("GHOSTFACENET_WEIGHTS_PATH", str(Path(default_root) / "GhostFaceNet.h5")),
        help="Path to local weights file (.h5). Only used if --use-local-weights is specified"
    )
    parser.add_argument(
        "--use-local-weights",
        action="store_true",
        help="Use local weights file instead of downloading via project loader"
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

    try:
        convert_ghostfacenet_to_savedmodel(
            export_dir=export_dir,
            weights_path=(Path(args.weights_path) if args.use_local_weights else None),
            use_load_model=not args.use_local_weights,
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
