""" 
Utilities for verifying converted models in DeepFace.
"""

# Import Required Packages
import onnx
import onnxruntime as ort
import numpy as np
import torch
from onnx2torch import convert


def confirm_graph_layout_onnx(model_name: str) -> None:
    """
    Confirm the layout of the ONNX model graph.
    
    Args:
        model_name (str): Path to the ONNX model file.
        
    Returns:
        None: Prints the first few nodes and Transpose perm if found.
    """
    model = onnx.load(model_name)
    graph = model.graph
    for n in graph.node[:3]:
        if n.op_type == "Transpose":
            print("front Transpose perm:", [a.ints for a in n.attribute if a.name=="perm"][0])
            
            
def check_input_shape(onnx_path: str) -> bool:
    """
    Check the input shape of the ONNX model to see if it expects NHWC or NCHW format.
    This function detects if the model graph expects NHWC input by checking for a front
    Transpose with perm=[0,3,1,2] applied to the graph input.
    
    Args:
        onnx_path (str): Path to the ONNX model file.
        
    Returns:
        None: Prints the input shape of the ONNX model.
    """
    expects_nhwc = False
    model = onnx.load(onnx_path)
    graph = model.graph
    if graph.node and graph.input:
        input_name = graph.input[0].name
        for n in graph.node[:3]:
            if n.op_type == "Transpose" and n.input and n.input[0] == input_name:
                for a in n.attribute:
                    if a.name == "perm" and list(a.ints) == [0,3,1,2]:
                        expects_nhwc = True
                        break
    print("expects_nhwc:", expects_nhwc)
    return expects_nhwc


def check_torch_equivalence(onnx_path: str, input_shape: tuple) -> None:
    """
    Check the equivalence of ONNX and PyTorch models by computing the error between their outputs.
    This function is robust if the ONNX model expects NHWC or NCHW input (this consistency can sometimes happen during conversion).
    
    Args:
        onnx_path (str): Path to the ONNX model file.
        input_shape (tuple): Shape of the input tensor.
        
    Returns:
        None: Prints the mean absolute error between ONNX and PyTorch outputs.
    """
    model = onnx.load(onnx_path)
    expects_nhwc = check_input_shape(onnx_path)
    pt_model = convert(model).eval()
    x_random = np.random.randn(1, 112, 112, 3).astype(np.float32)
    
    # Obtain reference output from ONNX Runtime implementation of model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ort_in = x_random if expects_nhwc else np.transpose(x_random, (0,3,1,2))
    ort_out = sess.run(None, {in_name: ort_in})[0]

    # Obtain PyTorch output, check to see if it matches ONNX output
    if expects_nhwc:
        x_pt = torch.from_numpy(x_random).contiguous()
    else:
        x_pt = torch.from_numpy(x_random).permute(0,3,1,2).contiguous()
    with torch.no_grad():
        y_pt = pt_model(x_pt).cpu().numpy()
        
    mae = np.mean(np.abs(y_pt - ort_out))
    print("mean abs diff:", mae)
    