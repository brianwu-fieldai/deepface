# deepface/models/ArcFaceTorch.py
from typing import List, Union, Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# ----------------------------
# Backbone: ResNet34 (IR-style) with PReLU
# ----------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample: Optional[nn.Module]=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(num_parameters=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = F.pad(out, (1,1,1,1), mode="constant", value=0)  # ZeroPadding2D equivalent before conv
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)

        out = F.pad(out, (1,1,1,1), mode="constant", value=0)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return out


class ResNet34IR(nn.Module):
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        # Stem
        self.conv1_pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.prelu1 = nn.PReLU(num_parameters=64)

        # Stacks
        self.layer2 = self._make_layer(64, 64, blocks=3, stride=2, name="conv2")
        self.layer3 = self._make_layer(64, 128, blocks=4, stride=2, name="conv3")
        self.layer4 = self._make_layer(128, 256, blocks=6, stride=2, name="conv4")
        self.layer5 = self._make_layer(256, 512, blocks=3, stride=2, name="conv5")

        # Head -> BN, Dropout, Flatten, Dense(512), BN (scale=True)
        self.head_bn = nn.BatchNorm2d(512, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)  # with 112x112 input, final map is 7x7
        self.out_bn = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)

        self._init()

    def _init(self):
        # Glorot/Xavier normal initializations to mirror Keras `glorot_normal`
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, in_planes, planes, blocks, stride, name):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9),
            )

        layers = [BasicBlock(in_planes, planes, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1_pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        # Stacks
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Head
        x = self.head_bn(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out_bn(x)
        return x


# ----------------------------
# Client that conforms to your ABC
# ----------------------------

class ArcFaceTorchClient(FacialRecognition):
    """
    ArcFace (PyTorch) model class that conforms to your FacialRecognition ABC.
    - Uses 112x112 RGB input
    - Produces 512-D embeddings
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        # device handling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # build model
        self.model = ResNet34IR(embedding_size=512).to(self._device)
        self.model.eval()

        # (Optional) load weights if provided
        if weights_path is not None:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            ckpt = torch.load(weights_path, map_location=self._device)
            # support either plain state_dict or wrapped dicts (e.g., {'state_dict': ...})
            state = ckpt.get("state_dict", ckpt)
            # strip potential "module." from DataParallel
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warn(f"Loaded weights with missing keys: {missing} and unexpected keys: {unexpected}")

        self.model_name = "ArcFaceTorch"
        self.input_shape: Tuple[int, int] = (112, 112)
        self.output_shape: int = 512

    # --- helpers ---

    @staticmethod
    def _ensure_nhwc(img: np.ndarray) -> np.ndarray:
        """
        Accepts (H, W, 3) or (N, H, W, 3). Returns (N, H, W, 3).
        """
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"Input must be (H, W, 3) or (N, H, W, 3), but got {img.shape}")
        return img

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        ArcFace-style preprocessing:
        - expects 112x112 RGB
        - convert to float32
        - normalize to [-1, 1] via (x - 127.5)/128
        - transpose NHWC -> NCHW
        """
        img = self._ensure_nhwc(img)
        if img.shape[1:3] != self.input_shape:
            raise ValueError(f"Expected spatial size {self.input_shape}, but got {img.shape[1:3]}")
        x = img.astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        t = torch.from_numpy(x).to(self._device)
        return t

    # --- ABC override ---

    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        """
        Accepts np.ndarray image(s) shaped (H, W, 3) or (N, H, W, 3).
        Returns Python lists per your ABC:
          - single image -> List[float] of length 512
          - batch -> List[List[float]] size [N, 512]
        """
        with torch.inference_mode():
            t = self._preprocess(img)
            emb = self.model(t)                 # [N, 512]
            emb = F.normalize(emb, p=2, dim=1)  # optional: L2 normalize embeddings
            out = emb.detach().cpu().numpy()
        if out.shape[0] == 1:
            return out[0].tolist()
        return out.tolist()
