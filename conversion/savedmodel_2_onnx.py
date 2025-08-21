# Import Required Packages
import tf2onnx
import subprocess
import argparse


def convert_savedmodel_to_onnx(saved_model_path: str, output_onnx_path: str, opset: int = 17) -> None:
    """
    Convert a TensorFlow SavedModel to ONNX format using a specified opset.

    Args:
        saved_model_path (str): Path to the input SavedModel directory.
        output_onnx_path (str): Path to save the converted ONNX model.
        opset (int): ONNX opset version to use for conversion.

    Returns:
        None: Saves the converted ONNX model to the specified path.
    """
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_path,
        "--opset", str(opset),
        "--output", output_onnx_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Converted SavedModel saved to {output_onnx_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to ONNX format")
    parser.add_argument("saved_model_path", type=str, help="Path to the input SavedModel directory")
    parser.add_argument("output_onnx_path", type=str, help="Path to save the converted ONNX model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version to use for conversion")
    args = parser.parse_args()
    convert_savedmodel_to_onnx(args.saved_model_path, args.output_onnx_path, args.opset)
    print("Conversion completed successfully.")
    print(f"ONNX model saved to {args.output_onnx_path}")
