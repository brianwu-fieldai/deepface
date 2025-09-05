from typing import Union, List, Any
from abc import ABC, abstractmethod
import numpy as np

# Notice that all facial attribute analysis models must be inherited from this class


# pylint: disable=too-few-public-methods
class Demography(ABC):
    model: Any  # Can be PyTorch model, ONNX session, or any other model type
    model_name: str

    @abstractmethod
    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, np.float64]:
        """
        Abstract method for running inference on input image(s).
        Each model implementation must override this method to handle
        their specific inference logic (PyTorch, ONNX, etc.).
        """
        pass

    @abstractmethod
    def _run_inference(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Abstract method for running model inference.
        Each implementation should handle their specific model type.
        
        Args:
            img_batch: 4D numpy array (n, h, w, c) where n >= 1
            
        Returns:
            Model predictions as numpy array
        """
        pass

    def _predict_internal(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Predict for single image or batched images.
        This method delegates to the abstract _run_inference method
        that each implementation must override.

        Args:
            img_batch:
                Batch of images as np.ndarray (n, x, y, c)
                    with n >= 1, x = image width, y = image height, c = channel
                Or Single image as np.ndarray (1, x, y, c)
                    with x = image width, y = image height and c = channel
                The channel dimension will be 1 if input is grayscale. (For emotion model)
        """
        if not self.model_name:  # Check if called from derived class
            raise NotImplementedError("no model selected")
        assert img_batch.ndim == 4, "expected 4-dimensional tensor input"

        # Delegate to implementation-specific inference method
        return self._run_inference(img_batch)

    def _preprocess_batch_or_single_input(
        self, img: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Preprocess single or batch of images, return as 4-D numpy array.
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                 List of images as List[np.ndarray] or
                 Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            Four-dimensional numpy array (n, 224, 224, 3)
        """
        image_batch = np.array(img)

        # Check input dimension
        if len(image_batch.shape) == 3:
            # Single image - add batch dimension
            image_batch = np.expand_dims(image_batch, axis=0)
        return image_batch
