from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import onnx
from onnx2torch import convert

# Conform to DeepFace's expected interface
from deepface.models.FacialRecognition import FacialRecognition

# --- CONFIG -------------------------------------------------------------------

GHOSTFACENET_ONNX_PATH = Path(
    "/Users/brianwu/Documents/Projects/FieldAI/deepface/conversion/onnx_models/GhostFaceNet.onnx"
)

# -----------------------------------------------------------------------------

def pick_device(prefer_mps: bool = True) -> torch.device:
    """
    Choose best available device. Prioritize CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


class GhostFaceNetTorchONNXClient(FacialRecognition):
    """
    GhostFaceNet client implemented in PyTorch via onnx2torch, while
    matching DeepFace's FacialRecognition interface.

    - model_name: "GhostFaceNetTorchONNX"
    - input_shape: (112, 112)
    - output_shape: 512
    """

    def __init__(
        self,
        onnx_path: Union[str, Path] = GHOSTFACENET_ONNX_PATH,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        expects_nhwc: Optional[bool] = None,  # if None, auto-detect from ONNX
    ):
        self.model_name = "GhostFaceNetTorchONNX"
        self.input_shape: Tuple[int, int] = (112, 112)
        self.output_shape: int = 512

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        self.device = device or pick_device()
        self.dtype = dtype

        # Load ONNX and convert to a PyTorch nn.Module
        onnx_model = onnx.load(self.onnx_path)
        self._expects_nhwc = (
            expects_nhwc if expects_nhwc is not None else self._detect_nhwc(onnx_model)
        )

        core = convert(onnx_model)
        self.model = core.to(self.device).to(self.dtype).eval()  # keep attribute 'model' per ABC

        # no gradients for inference
        torch.set_grad_enabled(False)

    # ----------------------------- helpers ------------------------------------

    @staticmethod
    def _detect_nhwc(onnx_model: onnx.ModelProto) -> bool:
        """
        Detect if ONNX graph expects NHWC by checking for an early Transpose
        with perm=[0,3,1,2] on the graph input (common for TF-exported models).
        """
        g = onnx_model.graph
        if not g.node or not g.input:
            # Default to NHWC if uncertain (common for TF-origin models)
            return True

        input_name = g.input[0].name
        # Look at a few nodes near the front
        for n in g.node[:4]:
            if n.op_type == "Transpose" and n.input and n.input[0] == input_name:
                for a in n.attribute:
                    if a.name == "perm" and list(a.ints) == [0, 3, 1, 2]:
                        # NHWC input transposed to NCHW at graph start -> original expects NHWC
                        return True
        # Otherwise assume NCHW
        return False

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Accepts (H,W,3) or (N,H,W,3) or (N,3,H,W).
        Converts to the layout the underlying model expects (NHWC or NCHW).
        """
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)  # -> (1, H, W, 3)

        if img.ndim != 4:
            raise ValueError(
                f"Input image must be (H,W,3) or (N,H,W,3) or (N,3,H,W), got {img.shape}"
            )

        # Ensure float32 numpy
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        x = torch.from_numpy(img)  # (N, H, W, 3) or (N, 3, H, W)

        if self._expects_nhwc:
            # Model expects NHWC
            if x.shape[1] == 3:
                # Input is NCHW -> convert to NHWC
                x = x.permute(0, 2, 3, 1).contiguous()
        else:
            # Model expects NCHW
            if x.shape[-1] == 3:
                # Input is NHWC -> convert to NCHW
                x = x.permute(0, 3, 1, 2).contiguous()

        return x.to(self.device, dtype=self.dtype)

    # ---------------------------- interface -----------------------------------

    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        """
        DeepFace interface:
        - If a single image -> returns List[float] (512 dims)
        - If a batch -> returns List[List[float]]
        Assumes images are already resized to 112x112 and normalized as you prefer.
        """
        x = self._to_tensor(img)

        # Validate spatial size
        if self._expects_nhwc:
            _, h, w, c = x.shape
            if (h, w, c) != (self.input_shape[0], self.input_shape[1], 3):
                raise ValueError(
                    f"Expected NHWC (H,W,C)={(self.input_shape[0], self.input_shape[1], 3)}, "
                    f"got {(h,w,c)}"
                )
        else:
            _, c, h, w = x.shape
            if (c, h, w) != (3, self.input_shape[0], self.input_shape[1]):
                raise ValueError(
                    f"Expected NCHW (C,H,W)={(3, self.input_shape[0], self.input_shape[1])}, "
                    f"got {(c,h,w)}"
                )

        with torch.no_grad():
            y = self.model(x)

            # Handle various output container types
            if isinstance(y, (list, tuple)):
                y = y[0]
            elif isinstance(y, dict):
                # Take the first output tensor deterministically
                y = next(iter(y.values()))

            y = y.detach().cpu().numpy()

        # Normalize return type to match ABC expectations
        if y.ndim == 1:
            # Already (512,)
            return y.tolist()
        if y.shape[0] == 1:
            return y[0].tolist()
        return y.tolist()

    # Optional: convenience wrapper (DeepFace sometimes calls .represent)
    def represent(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        return self.forward(img)


# ----------------------------- quick self-test --------------------------------
if __name__ == "__main__":
    # Smoke-test with a dummy image (values 0..1). Replace with real preprocessed image as needed.
    dummy = np.zeros((1, 112, 112, 3), dtype=np.float32)

    client = GhostFaceNetTorchONNXClient(onnx_path=GHOSTFACENET_ONNX_PATH)
    out = client.forward(dummy)

    if isinstance(out, list) and isinstance(out[0], float):
        print("OK: single embedding length:", len(out))
    else:
        print("OK: batch embeddings shape:", np.asarray(out).shape)
