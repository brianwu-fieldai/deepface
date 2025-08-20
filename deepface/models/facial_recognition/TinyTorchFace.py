# deepface/models/TinyTorchFace.py
from typing import List, Union, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepface.models.FacialRecognition import FacialRecognition

# ----- Tiny backbone -----
class TinyFaceNet(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.head  = nn.Linear(32, emb_dim)   # after global avg pool -> [N, 32]
        self.bn    = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W]
        x = F.relu(self.conv1(x))             # [N,16,H,W]
        x = F.max_pool2d(x, 2)                # [N,16,H/2,W/2]
        x = F.relu(self.conv2(x))             # [N,32,H/2,W/2]
        x = F.adaptive_avg_pool2d(x, 1)       # [N,32,1,1]
        x = x.view(x.size(0), 32)             # [N,32]
        x = self.head(x)                      # [N,emb_dim]
        x = self.bn(x)                        # [N,emb_dim]
        # L2-normalize (common for face embeddings; remove if undesired)
        x = F.normalize(x, p=2, dim=1)
        return x

# ----- Client that matches your ABC -----
class TinyTorchFaceClient(FacialRecognition):
    """
    Minimal PyTorch facial-recognition embedding model.
    - Input: RGB images, auto-resized to 64x64
    - Output: 128-D embedding(s) as Python list(s)
    """

    def __init__(self, device: Optional[str] = None):
        self.model_name = "TinyTorchFace"
        self.input_shape: Tuple[int, int] = (64, 64)  # nominal input shape
        self.output_shape: int = 128

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self.model = TinyFaceNet(emb_dim=self.output_shape).to(self._device)
        self.model.eval()

    # --- helpers ---
    @staticmethod
    def _ensure_nhwc(img: np.ndarray) -> np.ndarray:
        # Accept (H,W,3) or (N,H,W,3) and return (N,H,W,3)
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"Expected (H,W,3) or (N,H,W,3), got {img.shape}")
        return img

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        # Convert to float32 in [0,1], NHWC->NCHW, and resize to 64x64
        img = self._ensure_nhwc(img).astype(np.float32) / 255.0
        x = np.transpose(img, (0, 3, 1, 2))                 # [N,3,H,W]
        t = torch.from_numpy(x).to(self._device)
        t = F.interpolate(t, size=self.input_shape, mode="bilinear", align_corners=False)
        return t

    # --- ABC override ---
    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        with torch.inference_mode():
            t = self._preprocess(img)           # [N,3,64,64]
            emb = self.model(t)                 # [N,128]
            out = emb.detach().cpu().numpy()
        return out[0].tolist() if out.shape[0] == 1 else out.tolist()
