# Import Required Packages
import torch
from onnx2torch import convert
import onnx
import argparse


def convert_onnx_to_torch(input_model_path: str, output_model_path: str, print_model_tensors: bool = False) -> None:
    """
    Convert an ONNX model to a PyTorch model using onnx2torch.
    
    Args:
        input_model_path (str): Path to the input ONNX model file.
        output_model_path (str): Path to save the converted PyTorch model.
    """
    onnx_model = onnx.load(input_model_path)
    pt_model = convert(onnx_model)
    torch.save(pt_model.state_dict(), output_model_path)
    print(f"Converted ONNX model saved to {output_model_path}")
    if print_model_tensors:
        pt_model.eval()
        print("Model Tensors: ", [name for name, _ in pt_model.named_parameters()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to PyTorch model.")
    parser.add_argument("input_model", type=str, help="Path to the input ONNX model file.")
    parser.add_argument("output_model", type=str, help="Path to save the converted PyTorch model.")
    parser.add_argument("--print-tensors", action="store_true", help="Print model tensors after conversion.")

    args = parser.parse_args()
    convert_onnx_to_torch(args.input_model, args.output_model, args.print_tensors)
    