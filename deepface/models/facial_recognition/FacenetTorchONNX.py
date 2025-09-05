# project dependencies
from deepface.commons import package_utils, weight_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# -----------------------------
# Stdlib & typing
# -----------------------------
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

# runtime deps
import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("Please install onnxruntime (and onnxruntime-gpu if desired).") from e

# -----------------------------
# Defaults / Paths / Shapes
# -----------------------------
# Your local Facenet512 ONNX path (change if you move it)
DEFAULT_FACENET512_ONNX_PATH = Path(
    "/Users/brianwu/Documents/Projects/FieldAI/deepface/conversion/onnx_models/facenet512.onnx"
)

# Optional remote URL (leave empty if you only use the local file)
DEFAULT_ONNX_URL = os.getenv("FACENET512_ONNX_URL", "")

# Most TF/Keras exports are NHWC; keep True unless you know your ONNX expects NCHW
DEFAULT_CHANNELS_LAST_INPUT = os.getenv("FACENET512_CHANNELS_LAST", "0") not in ("0", "false", "False")

FACENET_INPUT_HW = (160, 160)  # (H, W)
FACENET_EMBED_DIM = 512        # output embedding dim

# -----------------------------
# ONNX Runtime helpers
# -----------------------------
def _select_ort_providers(prefer_gpu: bool = True) -> List[str]:
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _ensure_onnx_available(
    onnx_path: Optional[Union[str, Path]],
    onnx_url: Optional[str],
    cache_name: str = "facenet512.onnx",
) -> Path:
    """
    Resolve ONNX file path by preferring local path; otherwise download via URL (DeepFace cache).
    """
    if onnx_path:
        p = Path(onnx_path)
        if not p.exists():
            raise FileNotFoundError(f"ONNX file not found: {p}")
        return p

    url = onnx_url or DEFAULT_ONNX_URL
    if url:
        downloaded = weight_utils.download_weights_if_necessary(file_name=cache_name, source_url=url)
        return Path(downloaded)

    # Fall back to default local path
    if DEFAULT_FACENET512_ONNX_PATH.exists():
        return DEFAULT_FACENET512_ONNX_PATH

    raise ValueError(
        "No ONNX path provided, no URL configured, and default local path not found. "
        "Provide onnx_path or set FACENET512_ONNX_URL."
    )


# -----------------------------
# Torch wrapper around ORT session
# -----------------------------
class ONNXModule(nn.Module):
    """
    Thin torch.nn.Module wrapping an ONNXRuntime InferenceSession.

    Args:
        onnx_path: path to .onnx file
        prefer_gpu: try to use CUDA EP if available
        input_channels_last: whether *your input tensors* will be NHWC (True) or NCHW (False).
            NOTE: This wrapper feeds the ONNX as NHWC (typical TF->ONNX export).
    """
    def __init__(
        self,
        onnx_path: Union[str, Path],
        prefer_gpu: bool = True,
        input_channels_last: bool = False,
    ):
        super().__init__()
        providers = _select_ort_providers(prefer_gpu)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # we assume ONNX expects NHWC (channels_last) because it was exported from TF/Keras
        self.onnx_expects_channels_last = True
        self.input_channels_last = input_channels_last

        # just for logging/reference
        self.onnx_input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Loaded ONNX: {onnx_path} | providers={self.session.get_providers()} | "
                    f"input_name={self.input_name} | output_name={self.output_name} | "
                    f"shape={self.onnx_input_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch float32 tensor of shape:
           - if input_channels_last=True: (N, H, W, C)
           - else: (N, C, H, W)

        Returns:
           torch.float32 tensor of shape (N, 512)
        """
        if x.dtype != torch.float32:
            x = x.float()

        # Move to host memory for ORT
        x = x.detach().cpu()

        # Convert to NHWC for ONNX (if caller passed NCHW)
        if self.input_channels_last:
            if x.dim() != 4:
                raise ValueError("Expected 4D tensor (N,H,W,C) for input_channels_last=True")
            arr = x.numpy()  # already NHWC
        else:
            if x.dim() != 4:
                raise ValueError("Expected 4D tensor (N,C,H,W) for input_channels_last=False")
            arr = x.permute(0, 2, 3, 1).contiguous().numpy()  # NCHW -> NHWC

        out = self.session.run([self.output_name], {self.input_name: arr})[0]  # (N, 512)
        return torch.from_numpy(out)


def load_facenet512_onnx_model(
    onnx_path: Optional[Union[str, Path]] = None,
    onnx_url: Optional[str] = None,
    prefer_gpu: bool = True,
    input_channels_last: bool = DEFAULT_CHANNELS_LAST_INPUT,
) -> nn.Module:
    """
    Load FaceNet-512d ONNX as a torch.nn.Module (wrapped ORT).
    """
    model_file = _ensure_onnx_available(onnx_path, onnx_url, cache_name="facenet512.onnx")
    return ONNXModule(
        onnx_path=model_file,
        prefer_gpu=prefer_gpu,
        input_channels_last=input_channels_last,
    )


# -----------------------------
# Client class (inherits FacialRecognition)
# -----------------------------
class FaceNetTorchONNXClient(FacialRecognition):
    """
    FaceNet-512d model class (ONNX + PyTorch wrapper, no TensorFlow).
    """
    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        onnx_url: Optional[str] = None,
        prefer_gpu: bool = True,
        input_channels_last: bool = DEFAULT_CHANNELS_LAST_INPUT,
    ):
        self.model = load_facenet512_onnx_model(
            onnx_path=onnx_path,
            onnx_url=onnx_url,
            prefer_gpu=prefer_gpu,
            input_channels_last=input_channels_last,
        )
        self.model_name = "FaceNet-512d"
        self.input_shape = (160, 160)  # (W, H) convention in DeepFace clients
        self.output_shape = FACENET_EMBED_DIM


# -----------------------------
# Convenience preprocessing
# -----------------------------
def preprocess_to_tensor(
    img: np.ndarray,
    size_hw: Tuple[int, int] = FACENET_INPUT_HW,
    input_channels_last: bool = DEFAULT_CHANNELS_LAST_INPUT,
    scale01_if_needed: bool = True,
    prewhiten: bool = False,
) -> torch.Tensor:
    """
    Convert HxWxC RGB array to a batched tensor for the ONNX wrapper.

    Returns:
       - if input_channels_last=True: (1, H, W, C)
       - else: (1, C, H, W)
    """
    import cv2

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    H, W = size_hw
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    if scale01_if_needed and img_resized.max() > 1.5:
        img_resized = img_resized / 255.0

    if prewhiten:
        # Standard FaceNet prewhitening
        mean = np.mean(img_resized)
        std = np.std(img_resized)
        std_adj = np.maximum(std, 1.0 / np.sqrt(img_resized.size))
        img_resized = (img_resized - mean) / std_adj

    if input_channels_last:
        tensor = torch.from_numpy(img_resized).unsqueeze(0)  # (1,H,W,C)
    else:
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)

    return tensor.contiguous().float()


# -----------------------------
# Optional CLI test
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FaceNet-512d ONNX with PyTorch-style API")
    parser.add_argument("--onnx-path", type=str, help="Local path to facenet512.onnx")
    parser.add_argument("--onnx-url", type=str, help="URL to download facenet512.onnx")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no CUDA EP)")
    parser.add_argument("--channels-last-input", action="store_true",
                        help="Assume your input tensors are NHWC (default from env FACENET512_CHANNELS_LAST)")
    parser.add_argument("--image", type=str, help="Path to an RGB image to embed")
    args = parser.parse_args()

    client = FaceNetTorchONNXClient(
        onnx_path=args.onnx_path or str(DEFAULT_FACENET512_ONNX_PATH),
        onnx_url=args.onnx_url,
        prefer_gpu=not args.cpu,
        input_channels_last=args.channels_last_input or DEFAULT_CHANNELS_LAST_INPUT,
    )

    logger.info(f"Loaded {client.model_name}; providers: {client.model.session.get_providers()}")
    if args.image:
        import cv2
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        x = preprocess_to_tensor(
            rgb,
            size_hw=FACENET_INPUT_HW,
            input_channels_last=(client.model.input_channels_last),
            scale01_if_needed=True,
            prewhiten=False,
        )

        with torch.no_grad():
            emb = client.model(x)  # (1, 512)
        print("Embedding shape:", tuple(emb.shape))
        print("Embedding (first 5):", emb[0, :5].numpy())
