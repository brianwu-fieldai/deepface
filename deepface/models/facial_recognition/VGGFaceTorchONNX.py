from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import onnx
from onnx2torch import convert

# Conform to DeepFace's expected interface
from deepface.models.FacialRecognition import FacialRecognition

# Default ONNX path (adjust if needed)
WEIGHTS_PATH = Path("/Users/brianwu/Documents/Projects/FieldAI/deepface/conversion/onnx_models/vggface.onnx")


def pick_device(prefer_mps: bool = True) -> torch.device:
    """
    Choose best available device. Prioritize CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Simple L2 normalization in NumPy to avoid TF/Keras deps.
    """
    denom = np.sqrt(np.maximum((x ** 2).sum(axis=axis, keepdims=True), eps))
    return x / denom


class VggFaceTorchONNXClient(FacialRecognition):
    """
    VGG-Face client using PyTorch and onnx2torch under the hood,
    matching DeepFace's FacialRecognition interface.

    - model_name: "VGG-FaceTorchONNX"
    - input_shape: (224, 224)
    - output_shape: 4096
    """

    def __init__(
        self,
        onnx_path: Union[str, Path] = WEIGHTS_PATH,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        expects_nhwc: Optional[bool] = None,   # if None, auto-detect from ONNX
        normalize: bool = True,                # match existing VggFaceClient.forward() behavior
    ):
        self.model_name = "VGG-FaceTorchONNX"
        self.input_shape: Tuple[int, int] = (224, 224)
        self.output_shape: int = 4096

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        self.device = device or pick_device()
        self.dtype = dtype
        self.normalize = normalize

        # Load ONNX, detect layout, convert to torch.nn.Module
        onnx_model = onnx.load(self.onnx_path)
        self._expects_nhwc = (
            expects_nhwc if expects_nhwc is not None else self._detect_nhwc(onnx_model)
        )

        core = convert(onnx_model)
        self.model = core.to(self.device).to(self.dtype).eval()

        # Disable autograd for inference
        torch.set_grad_enabled(False)

    # ----------------------------- helpers ------------------------------------

    @staticmethod
    def _detect_nhwc(onnx_model: onnx.ModelProto) -> bool:
        """
        Detect if ONNX graph expects NHWC by checking for an early Transpose
        with perm=[0,3,1,2] on the graph input (typical for TF exports).
        """
        g = onnx_model.graph
        if not g.node or not g.input:
            return True  # default to NHWC if uncertain
        input_name = g.input[0].name
        for n in g.node[:4]:
            if n.op_type == "Transpose" and n.input and n.input[0] == input_name:
                for a in n.attribute:
                    if a.name == "perm" and list(a.ints) == [0, 3, 1, 2]:
                        return True
        return False  # otherwise assume NCHW

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

        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        x = torch.from_numpy(img)  # (N, H, W, 3) or (N, 3, H, W)

        if self._expects_nhwc:
            # Model expects NHWC
            if x.shape[1] == 3:
                x = x.permute(0, 2, 3, 1).contiguous()
        else:
            # Model expects NCHW
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2).contiguous()

        return x.to(self.device, dtype=self.dtype)

    # ---------------------------- interface -----------------------------------

    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        """
        DeepFace interface:
        - If a single image -> returns List[float] (4096 dims)
        - If a batch -> returns List[List[float]]

        Assumes images are already resized to 224x224 and preprocessed (BGR/RGB, scaling)
        as required by your pipeline.
        """
        x = self._to_tensor(img)

        # Validate spatial size
        if self._expects_nhwc:
            _, h, w, c = x.shape
            if (h, w, c) != (self.input_shape[0], self.input_shape[1], 3):
                raise ValueError(
                    f"Expected NHWC (H,W,C)={(self.input_shape[0], self.input_shape[1], 3)}, got {(h,w,c)}"
                )
        else:
            _, c, h, w = x.shape
            if (c, h, w) != (3, self.input_shape[0], self.input_shape[1]):
                raise ValueError(
                    f"Expected NCHW (C,H,W)={(3, self.input_shape[0], self.input_shape[1])}, got {(c,h,w)}"
                )

        with torch.no_grad():
            y = self.model(x)
            # Normalize possible container outputs
            if isinstance(y, (list, tuple)):
                y = y[0]
            elif isinstance(y, dict):
                y = next(iter(y.values()))

            y = y.detach().cpu().numpy()

        # Robustly flatten to (N, D)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        elif y.ndim > 2:
            y = y.reshape(y.shape[0], -1)

        # Optional L2 normalization (matches existing VggFaceClient.forward)
        if self.normalize:
            y = _l2_normalize(y, axis=1)

        # Convert to DeepFace's expected return types
        if y.shape[0] == 1:
            return y[0].tolist()
        return y.tolist()

    # Optional convenience alias
    def represent(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        return self.forward(img)


# ----------------------------- quick self-test --------------------------------
if __name__ == "__main__":
    # Zero image smoke-test
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    client = VggFaceTorchONNXClient(onnx_path=WEIGHTS_PATH)
    out = client.forward(dummy)
    arr = np.asarray(out)
    if arr.ndim == 1:
        print("OK: single embedding length:", arr.shape[0])
    else:
        print("OK: batch embeddings shape:", arr.shape)
