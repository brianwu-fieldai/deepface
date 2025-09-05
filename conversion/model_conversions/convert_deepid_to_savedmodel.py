"""
DeepID model conversion utilities.

This module provides functions to convert DeepID models to SavedModel format
with options to build from scratch or use the existing load_model function.
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
    from keras.layers import (
        Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout
    )
else:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout
    )

# If you have these in your project (as in your snippet), we can reuse them:
from deepface.commons import weight_utils
from deepface.models.FacialRecognition import FacialRecognition  # noqa: F401  # for parity with project layout

# ---- Defaults (same as your project) ----
WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5"


def build_deepid_model() -> Model:
    """
    Build DeepID model architecture from scratch (no weights).
    Input: (55, 47, 3)
    Output: 160-dim "deepid" embedding (ReLU)
    """
    myInput = Input(shape=(55, 47, 3))

    x = Conv2D(20, (4, 4), name="Conv1", activation="relu", input_shape=(55, 47, 3))(myInput)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
    x = Dropout(rate=0.99, name="D1")(x)

    x = Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
    x = Dropout(rate=0.99, name="D2")(x)

    x = Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
    x = Dropout(rate=0.99, name="D3")(x)

    x1 = Flatten()(x)
    fc11 = Dense(160, name="fc11")(x1)

    x2 = Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name="fc12")(x2)

    y = Add()([fc11, fc12])
    y = Activation("relu", name="deepid")(y)

    model = Model(inputs=[myInput], outputs=y, name="DeepID")
    return model


def load_model_via_project(url: Optional[str] = None) -> Model:
    """
    Use the same download + weight loading logic your project uses.
    """
    target_url = url or WEIGHTS_URL

    model = build_deepid_model()

    # Download (or reuse cached) weights
    weight_file = weight_utils.download_weights_if_necessary(
        file_name="deepid_keras_weights.h5", source_url=target_url
    )

    # Load weights (by layer names)
    model = weight_utils.load_model_weights(model=model, weight_file=weight_file)
    return model


def convert_deepid_to_savedmodel(
    export_dir: Union[str, Path],
    weights_path: Optional[Union[str, Path]] = None,
    use_load_model: bool = False,
    weights_url: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Convert DeepID to SavedModel format.

    Args:
        export_dir: Directory to save the SavedModel
        weights_path: Local .h5 weights path (used only if use_load_model=False)
        use_load_model: If True, download/load with project logic; else build + load local weights
        weights_url: Optional custom URL used when use_load_model=True
        verbose: Print progress

    Returns:
        str: absolute path to SavedModel directory
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Export directory: {export_dir.resolve()}")

    if use_load_model:
        if verbose:
            print("Using project load flow (will download weights if needed).")
        model = load_model_via_project(weights_url)
        if verbose:
            print("Model constructed and weights loaded via project flow.")
    else:
        if weights_path is None:
            raise ValueError("weights_path must be provided when use_load_model=False")

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        if verbose:
            print(f"Building model from scratch and loading weights from: {weights_path}")

        model = build_deepid_model()
        # Load exact Keras H5 weights by name (deepid h5 uses named layers)
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=False)

        if verbose:
            print("Weights loaded successfully.")

    # Build graph with a dummy input (NHWC: (1, 55, 47, 3))
    dummy = np.zeros((1, 55, 47, 3), dtype="float32")
    _ = model(dummy, training=False)
    if verbose:
        print("Model graph built.")

    # Export SavedModel (Keras 3 API). Falls back if unavailable.
    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        # For older TF/Keras, this is equivalent
        model.save(str(export_dir), save_format="tf")
    if verbose:
        print(f"SavedModel written to: {export_dir}")

    return str(export_dir.resolve())


def main():
    """
    CLI parity with your ArcFace script.
    """
    parser = argparse.ArgumentParser(
        description="Convert DeepID model to SavedModel format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_root = os.getenv("DEEPID_CONVERSION_ROOT", Path.cwd())
    default_weights = os.getenv(
        "DEEPID_WEIGHTS_PATH",
        str(Path(default_root) / "deepid_keras_weights.h5")
    )
    default_export = os.getenv(
        "DEEPID_EXPORT_DIR",
        str(Path(default_root) / "deepid_saved_model")
    )

    parser.add_argument(
        "--export-dir", "-o", type=str, default=default_export,
        help="Directory to save the SavedModel directory"
    )
    parser.add_argument(
        "--weights-path", "-w", type=str, default=default_weights,
        help="Path to local weights file (.h5). Only used if --use-local-weights is specified"
    )
    parser.add_argument(
        "--use-local-weights",
        action="store_true",
        help="Use a local .h5 file instead of downloading via project loader"
    )
    parser.add_argument(
        "--weights-url",
        type=str,
        help="Custom weights URL to use with project loader (optional)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    weights_path = Path(args.weights_path) if args.use_local_weights else None

    if args.use_local_weights:
        if not weights_path.exists():
            print(f"Warning: Weights file not found: {weights_path}")
            resp = input("Continue anyway? The function will raise an error. (y/N): ")
            if resp.lower() != "y":
                print("Aborted.")
                return
        else:
            print(f"Weights file found: {weights_path}")

    try:
        convert_deepid_to_savedmodel(
            export_dir=export_dir,
            weights_path=weights_path,
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
