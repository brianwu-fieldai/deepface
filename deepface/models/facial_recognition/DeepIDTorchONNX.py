# project dependencies
from deepface.commons import package_utils, weight_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

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
# Defaults / Env
# -----------------------------
# Optional default URL for your DeepID .onnx (replace with your hosted model if you have one)
DEFAULT_ONNX_URL = os.getenv("DEEPID_ONNX_URL", "")
# If your ONNX was converted from Keras with channels_last, keep this True.
# If you know the ONNX expects NCHW, set to False.
DEFAULT_CHANNELS_LAST = os.getenv("DEEPID_CHANNELS_LAST", "1") not in ("0", "false", "False")

# DeepID expects input size (H, W) = (55, 47), 3 channels (RGB)
DEEPID_INPUT_HW = (55, 47)
DEEPID_EMBED_DIM = 160


def _select_ort_providers(prefer_gpu: bool = True) -> List[str]:
    """
    Choose ONNX Runtime execution providers.
    """
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ONNXModule(nn.Module):
    """
    A thin torch.nn.Module wrapper around an ONNXRuntime InferenceSession.

    Args:
        onnx_path: path to .onnx file
        prefer_gpu: try to use CUDA EP if available
        channels_last: whether input ONNX expects NHWC (True) or NCHW (False)
    """
    def __init__(self, onnx_path: Union[str, Path], prefer_gpu: bool = True, channels_last: bool = True):
        super().__init__()
        onnx_path = str(onnx_path)
        providers = _select_ort_providers(prefer_gpu)
        sess_options = ort.SessionOptions()
        # mild graph optimizations
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.channels_last = channels_last

        # record expected shape from ONNX (N, C/H, H/W, W/C)
        self.onnx_input_shape = self.session.get_inputs()[0].shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch tensor of shape:
           - if channels_last=True: (N, H, W, C)
           - else: (N, C, H, W)

        Returns:
           torch float32 tensor of shape (N, 160)
        """
        if x.dtype != torch.float32:
            x = x.float()

        # Move to CPU for ORT (ORT takes numpy on host)
        if x.device.type != "cpu":
            x = x.detach().cpu()
        else:
            x = x.detach()

        # Convert channel order to match ONNX expectation
        if self.channels_last:
            # ensure NHWC
            if x.dim() != 4:
                raise ValueError("Expected 4D tensor (N,H,W,C) for channels_last=True")
            arr = x.numpy()
        else:
            # ensure NCHW -> NHWC for models that expect NHWC
            if x.dim() != 4:
                raise ValueError("Expected 4D tensor (N,C,H,W) for channels_last=False")
            arr = x.permute(0, 2, 3, 1).numpy()  # NCHW -> NHWC

        # Run ONNX
        out = self.session.run([self.output_name], {self.input_name: arr})[0]  # (N, 160)
        return torch.from_numpy(out)


def _ensure_onnx_available(
    onnx_path: Optional[Union[str, Path]],
    onnx_url: Optional[str],
    cache_name: str = "deepid.onnx",
) -> Path:
    """
    Resolve ONNX file path by preferring local path; otherwise download via URL.
    """
    if onnx_path:
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        return onnx_path

    url = onnx_url or DEFAULT_ONNX_URL
    if not url:
        raise ValueError(
            "No ONNX path provided and no ONNX URL configured. "
            "Pass onnx_path or set --onnx-url / DEEPID_ONNX_URL."
        )

    # Reuse deepface weight_utils cache
    downloaded = weight_utils.download_weights_if_necessary(file_name=cache_name, source_url=url)
    return Path(downloaded)


def load_model(
    onnx_path: Optional[Union[str, Path]] = None,
    onnx_url: Optional[str] = None,
    prefer_gpu: bool = True,
    channels_last: bool = DEFAULT_CHANNELS_LAST,
) -> nn.Module:
    """
    Load DeepID ONNX as a PyTorch nn.Module (wrapped with ONNXRuntime).
    """
    onnx_file = _ensure_onnx_available(onnx_path, onnx_url, cache_name="deepid.onnx")
    model = ONNXModule(onnx_file, prefer_gpu=prefer_gpu, channels_last=channels_last)
    return model


# -----------------------------
# DeepIdClient (PyTorch)
# -----------------------------
class DeepIdTorchONNXClient(FacialRecognition):
    """
    DeepId model class using ONNX + PyTorch-style module.
    """
    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        onnx_url: Optional[str] = None,
        prefer_gpu: bool = True,
        channels_last: bool = DEFAULT_CHANNELS_LAST,
    ):
        self.model = load_model(
            onnx_path=onnx_path,
            onnx_url=onnx_url,
            prefer_gpu=prefer_gpu,
            channels_last=channels_last,
        )
        self.model_name = "DeepId"
        # Keep parity with your prior client (note prior code listed (47, 55); the network expects (55, 47))
        self.input_shape = (47, 55)  # (W, H) for convenience if you use this downstream
        self.output_shape = DEEPID_EMBED_DIM


# -----------------------------
# Convenience: preprocessing & inference helpers
# -----------------------------
def preprocess_to_tensor(
    img: np.ndarray,
    size_hw: Tuple[int, int] = DEEPID_INPUT_HW,
    channels_last: bool = DEFAULT_CHANNELS_LAST,
) -> torch.Tensor:
    """
    img: HxWxC uint8 or float32, RGB. Resized to (55,47).
    Returns a batched float32 tensor ready for model.forward():
       - if channels_last=True: (1, H, W, C)
       - else: (1, C, H, W)
    """
    import cv2

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    # Resize to model size
    H, W = size_hw
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalize if needed (DeepID training typically used raw or simple scaling; adapt as required)
    # Here we scale to [0,1]
    img_resized = img_resized / 255.0

    if channels_last:
        tensor = torch.from_numpy(img_resized).unsqueeze(0)  # (1,H,W,C)
    else:
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)

    return tensor.contiguous().float()


# -----------------------------
# Optional CLI for quick tests
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run DeepID ONNX with PyTorch-style API"
    )
    parser.add_argument("--onnx-path", type=str, help="Local path to DeepID .onnx")
    parser.add_argument("--onnx-url", type=str, help="URL to download DeepID .onnx")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no CUDA EP)")
    parser.add_argument("--channels-last", action="store_true",
                        help="Assume ONNX expects NHWC (set if converted from Keras)")
    parser.add_argument("--image", type=str, help="Path to an RGB image to embed")
    args = parser.parse_args()

    client = DeepIdTorchONNXClient(
        onnx_path=args.onnx_path,
        onnx_url=args.onnx_url,
        prefer_gpu=not args.cpu,
        channels_last=args.channels_last or DEFAULT_CHANNELS_LAST,
    )

    print(f"Loaded {client.model_name} ONNX; providers: {client.model.session.get_providers()}")
    if args.image:
        import cv2
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        x = preprocess_to_tensor(rgb, size_hw=DEEPID_INPUT_HW,
                                 channels_last=(client.model.channels_last))
        with torch.no_grad():
            emb = client.model(x)  # (1, 160)
        print("Embedding shape:", tuple(emb.shape))
        print("Embedding (first 5):", emb[0, :5].numpy())
