"""
ArcFace model conversion utilities.

This module provides functions to convert ArcFace models to SavedModel format
with options to build from scratch or use the existing load_model function.
"""

# Import Required Packages
from pathlib import Path
from typing import Union, Optional
import numpy as np
import argparse
import os
from deepface.commons import package_utils
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import BatchNormalization, Dropout, Flatten, Dense
else:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Dense

from deepface.models.facial_recognition.ArcFace import ResNet34, load_model


def build_arcface_model() -> Model:
    """
    Build ArcFace model architecture from scratch.
    
    Returns:
        Model: The constructed ArcFace model (without weights loaded)
    """
    base_model = ResNet34()
    inputs = base_model.inputs[0]
    x = base_model.outputs[0]
    x = BatchNormalization(momentum=0.9, epsilon=2e-5)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(x)
    embedding = BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(x)
    model = Model(inputs, embedding, name=base_model.name)
    return model


def convert_arcface_to_savedmodel(
    export_dir: Union[str, Path],
    weights_path: Optional[Union[str, Path]] = None,
    use_load_model: bool = False,
    weights_url: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Convert ArcFace model to SavedModel format.
    
    Args:
        export_dir: Directory to save the SavedModel
        weights_path: Path to local weights file (.h5). Used only if use_load_model=False
        use_load_model: If True, use the existing load_model function (downloads weights automatically).
                       If False, build model from scratch and load local weights
        weights_url: Custom weights URL to use with load_model (optional)
        verbose: Whether to print progress messages
        
    Returns:
        str: Path to the exported SavedModel directory
        
    Raises:
        ValueError: If use_load_model=False but weights_path is not provided or doesn't exist
        FileNotFoundError: If weights_path doesn't exist when use_load_model=False
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Export directory: {export_dir}")
    
    if use_load_model:
        if verbose:
            print("Using load_model function (will download weights if needed)")
        if weights_url:
            model = load_model(url=weights_url)
        else:
            model = load_model()
        if verbose:
            print("Model loaded successfully")
    else:
        if weights_path is None:
            raise ValueError("weights_path must be provided when use_load_model=False")
        
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        if verbose:
            print(f"Building model from scratch and loading weights from: {weights_path}")
        
        model = build_arcface_model()
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=False)
        
        if verbose:
            print("Weights loaded successfully")
    
    dummy = np.zeros((1, 112, 112, 3), dtype="float32")
    _ = model(dummy, training=False)
    
    if verbose:
        print("Model graph built")
    
    model.export(str(export_dir))
    
    if verbose:
        print(f"SavedModel written to: {export_dir}")
    
    return str(export_dir)


def main():
    """
    Main function for command-line usage with configurable paths.
    
    Supports command-line arguments and environment variables for configuration.
    """
    parser = argparse.ArgumentParser(
        description="Convert ArcFace model to SavedModel format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_root = os.getenv("ARCFACE_CONVERSION_ROOT", Path.cwd())
    default_weights = os.getenv("ARCFACE_WEIGHTS_PATH", str(Path(default_root) / "arcface_weights.h5"))
    default_export = os.getenv("ARCFACE_EXPORT_DIR", str(Path(default_root) / "arcface_saved_model"))
    
    parser.add_argument(
        "--export-dir", "-o",
        type=str,
        default=default_export,
        help="Directory to save the SavedModel directory"
    )
    parser.add_argument(
        "--weights-path", "-w",
        type=str,
        default=default_weights,
        help="Path to local weights file (.h5). Only used if --use-local-weights is specified"
    )
    parser.add_argument(
        "--use-local-weights",
        action="store_true",
        help="Use local weights file instead of downloading via load_model"
    )
    parser.add_argument(
        "--weights-url",
        type=str,
        help="Custom weights URL to use with load_model (optional)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    export_dir = Path(args.export_dir)
    weights_path = Path(args.weights_path) if args.use_local_weights else None
    if args.use_local_weights:
        weights_path = Path(args.weights_path)
        if not weights_path.exists():
            print(f"Warning: Weights file not found: {weights_path}")
            print(f"File exists: {weights_path.exists()}")
            response = input("Continue anyway? The function will raise an error. (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        else:
            print(f"Weights file found: {weights_path}")
    else:
        weights_path = None
    
    try:
        convert_arcface_to_savedmodel(
            export_dir=export_dir,
            weights_path=weights_path,
            use_load_model=not args.use_local_weights,
            weights_url=args.weights_url,
            verbose=not args.quiet
        )
        print(f"Conversion completed successfully!")
        print(f"SavedModel location: {export_dir}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
