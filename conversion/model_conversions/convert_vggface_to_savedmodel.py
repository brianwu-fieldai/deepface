"""
VGG-Face model conversion utilities.

Converts the VGG-Face Keras (.h5) weights to TensorFlow SavedModel format.
You can either:
- Build the model locally and load a given .h5 file, or
- Use the project's load_model (downloads weights if needed).

Output is the 4096-d embedding head (pre-normalization), matching DeepFace's
current VGG-Face descriptor implementation.
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
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )
else:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )

# Import only the build/load utilities we need from your existing VGGFace module
# (Assumes this file lives alongside that module path)
from deepface.models.facial_recognition.VGGFace import base_model as vgg_base_model  # if your module path differs, adjust
from deepface.models.facial_recognition.VGGFace import load_model as vgg_load_model  # uses download + returns descriptor


def build_vggface_descriptor_model() -> Model:
    """
    Build VGG-Face architecture (classification) and expose the 4096-d descriptor head,
    mirroring the project's load_model behavior (without downloading).
    """
    model: Sequential = vgg_base_model()

    # As in your current load_model(): take the pre-softmax 4096-d vector
    # model.layers[-5] is the Conv2D(4096, (1,1), activation="relu")
    # They build descriptor as Flatten() over that feature map.
    base_model_output = Flatten()(model.layers[-5].output)
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=base_model_output, name="VGGFaceDescriptor")
    return vgg_face_descriptor


def convert_vggface_to_savedmodel(
    export_dir: Union[str, Path],
    weights_path: Optional[Union[str, Path]] = None,
    use_load_model: bool = False,
    weights_url: Optional[str] = None,  # kept for parity; vgg_load_model(url=...) supports it
    verbose: bool = True,
) -> str:
    """
    Convert VGG-Face to SavedModel format.

    Args:
        export_dir: Directory to save the SavedModel.
        weights_path: Local .h5 path (used when use_load_model=False).
        use_load_model: If True, call vgg_load_model() (downloads if needed).
                        If False, build descriptor and load local weights.
        weights_url: Optional URL for vgg_load_model(url=...).
        verbose: Print progress.

    Returns:
        str: Absolute path to SavedModel directory.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Export directory: {export_dir.resolve()}")
        print("Model: VGG-Face (4096-d descriptor)")

    if use_load_model:
        if verbose:
            print("Using project loader (will download weights if needed).")
        model = vgg_load_model(url=weights_url) if weights_url else vgg_load_model()
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

        # Build the classification backbone (through base_model inside descriptor builder)
        # and expose the 4096-d descriptor head, same as your existing loader does.
        # IMPORTANT: weights must match the *classification* graph nodes.
        descriptor = build_vggface_descriptor_model()

        # Load weights into the underlying layers. Because descriptor's graph reuses
        # tensors from base_model layers (up to [-5]), by_name=True will map correctly.
        # skip_mismatch=False to catch any structure mismatches early.
        descriptor.load_weights(str(weights_path), by_name=True, skip_mismatch=False)
        model = descriptor

        if verbose:
            print("Weights loaded successfully.")

    # Warm up (VGG-Face expects 224x224 BGR in original pipeline, but tensor shape is (H,W,3))
    dummy = np.zeros((1, 224, 224, 3), dtype="float32")
    _ = model(dummy, training=False)

    if verbose:
        print("Model graph built.")

    # TF/Keras >= 2.14 has model.export; otherwise fall back to SavedModel save
    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        # Older TF/Keras
        model.save(str(export_dir), save_format="tf")

    if verbose:
        print(f"SavedModel written to: {export_dir.resolve()}")

    return str(export_dir.resolve())


def main():
    """
    CLI for converting VGG-Face to SavedModel.
    """
    parser = argparse.ArgumentParser(
        description="Convert VGG-Face model (.h5) to TensorFlow SavedModel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_root = os.getenv("VGGFACE_CONVERSION_ROOT", Path.cwd())
    default_export = os.getenv("VGGFACE_EXPORT_DIR", str(Path(default_root) / "vggface_saved_model"))
    default_weights = os.getenv("VGGFACE_WEIGHTS_PATH", str(Path(default_root) / "vggface.h5"))

    parser.add_argument(
        "--export-dir", "-o",
        type=str,
        default=default_export,
        help="Directory to save the SavedModel directory",
    )
    parser.add_argument(
        "--weights-path", "-w",
        type=str,
        default=default_weights,
        help="Path to local weights file (.h5). Only used if --use-local-weights is specified",
    )
    parser.add_argument(
        "--use-local-weights",
        action="store_true",
        help="Use local weights file instead of downloading via project loader",
    )
    parser.add_argument(
        "--weights-url",
        type=str,
        help="Custom weights URL (used only if not using local weights)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
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
        convert_vggface_to_savedmodel(
            export_dir=export_dir,
            weights_path=(Path(args.weights_path) if args.use_local_weights else None),
            use_load_model=not args.use_local_weights,
            weights_url=args.weights_url,
            verbose=not args.quiet,
        )
        print("Conversion completed successfully!")
        print(f"SavedModel location: {export_dir}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
