from abc import ABC, abstractmethod
from typing import Any, Union, List, Tuple
import numpy as np

# Notice that all facial recognition models must be inherited from this class


# pylint: disable=too-few-public-methods
class FacialRecognition(ABC):
    model: Any  # Can be any model type (PyTorch, ONNX, TensorFlow, etc.)
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int

    @abstractmethod
    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        """
        Abstract method for running inference on input image(s).
        Each model implementation must override this method.
        
        Args:
            img: Input image array, expects (H, W, C) or (N, H, W, C) format
            
        Returns:
            List of embeddings - single list for one image, list of lists for batch
        """
        pass
