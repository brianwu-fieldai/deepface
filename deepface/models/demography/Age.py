# stdlib dependencies

from typing import List, Union, Any

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.commons import weight_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

logger = Logger()

# Model weights URL - can be used for downloading pre-trained weights
WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5"
)


# pylint: disable=too-few-public-methods
class ApparentAgeClient(Demography):
    """
    Age model class - Framework agnostic implementation
    """

    def __init__(self, model: Any, model_type: str = "pytorch"):
        """
        Initialize age model
        Args:
            model: Pre-trained model (PyTorch model, ONNX session, etc.)
            model_type: Type of model ("pytorch", "onnx", etc.)
        """
        self.model = model
        self.model_name = "Age"
        self.model_type = model_type

    def _run_inference(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded model
        Args:
            img_batch: Batch of preprocessed images (N, H, W, C)
        Returns:
            Age predictions as numpy array
        """
        if self.model_type == "pytorch":
            return self._pytorch_inference(img_batch)
        elif self.model_type == "onnx":
            return self._onnx_inference(img_batch)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not supported")

    def _pytorch_inference(self, img_batch: np.ndarray) -> np.ndarray:
        """PyTorch inference implementation"""
        try:
            import torch
            
            # Convert to PyTorch tensor
            tensor_input = torch.from_numpy(img_batch).float()
            
            # Run inference
            with torch.no_grad():
                self.model.eval()
                output = self.model(tensor_input)
                
                # Apply softmax if not already applied
                if not hasattr(self.model, 'softmax_applied'):
                    output = torch.softmax(output, dim=1)
            
            return output.cpu().numpy()
            
        except ImportError:
            raise ImportError("PyTorch is required for pytorch model type")

    def _onnx_inference(self, img_batch: np.ndarray) -> np.ndarray:
        """ONNX inference implementation"""
        try:
            # Get input name from ONNX session
            input_name = self.model.get_inputs()[0].name
            
            # Run inference
            outputs = self.model.run(None, {input_name: img_batch.astype(np.float32)})
            return outputs[0]
            
        except AttributeError:
            raise ValueError("Model must be an ONNX InferenceSession for onnx model type")

    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> Union[np.float64, np.ndarray]:
        """
        Predict apparent age(s) for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.float64 if single image, np.ndarray if batched images.
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction from 3 channels image
        age_predictions = self._predict_internal(imgs)

        # Calculate apparent ages
        if len(age_predictions.shape) == 1:  # Single prediction list
            return find_apparent_age(age_predictions)

        return np.array([find_apparent_age(age_prediction) for age_prediction in age_predictions])


def create_pytorch_age_model(num_classes: int = 101) -> Any:
    """
    Create a PyTorch age model architecture
    Args:
        num_classes: Number of age classes (default 101 for ages 0-100)
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        
        class AgeModel(nn.Module):
            def __init__(self, num_classes=101):
                super(AgeModel, self).__init__()
                # This is a placeholder - you would implement your actual architecture
                # based on the VGGFace base model structure
                self.features = nn.Sequential(
                    # Add your feature extraction layers here
                    # This would be equivalent to the VGGFace base model
                )
                self.classifier = nn.Sequential(
                    nn.Conv2d(512, num_classes, kernel_size=1),  # Equivalent to Convolution2D
                    nn.Flatten(),
                    nn.Softmax(dim=1)  # Equivalent to Activation("softmax")
                )
                
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return AgeModel(num_classes)
        
    except ImportError:
        raise ImportError("PyTorch is required to create PyTorch age model")


def load_pytorch_model(model_path: Union[str, None] = None) -> Any:
    """
    Load a pre-trained PyTorch age model
    Args:
        model_path: Path to the saved PyTorch model
    Returns:
        Loaded PyTorch model
    """
    try:
        import torch
        
        if model_path:
            model = torch.load(model_path, map_location='cpu')
        else:
            # Create a new model - you would need to implement weight loading
            model = create_pytorch_age_model()
            logger.info("Created new PyTorch age model - weights need to be loaded separately")
            
        model.eval()
        return model
        
    except ImportError:
        raise ImportError("PyTorch is required to load PyTorch age model")


def load_onnx_model(model_path: str) -> Any:
    """
    Load an ONNX age model
    Args:
        model_path: Path to the ONNX model file
    Returns:
        ONNX InferenceSession
    """
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(model_path)
        return session
        
    except ImportError:
        raise ImportError("onnxruntime is required to load ONNX models")


def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (age_classes,)
    Returns:
        apparent_age (float)
    """
    assert (
        len(age_predictions.shape) == 1
    ), f"Input should be a list of predictions, \
                                             not batched. Got shape: {age_predictions.shape}"
    output_indexes = np.arange(0, 101)
    apparent_age = np.sum(age_predictions * output_indexes)
    return np.float64(apparent_age)
