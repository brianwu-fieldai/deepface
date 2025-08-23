# stdlib dependencies
from typing import List, Union, Any

# 3rd party dependencies
import numpy as np
import cv2

# project dependencies
from deepface.commons import weight_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

# Labels for the emotions that can be detected by the model.
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

logger = Logger()

# pylint: disable=line-too-long, disable=too-few-public-methods

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"


class EmotionClient(Demography):
    """
    Emotion model class - Framework agnostic implementation
    """

    def __init__(self, model: Any, model_type: str = "pytorch"):
        """
        Initialize emotion model
        Args:
            model: Pre-trained model (PyTorch model, ONNX session, etc.)
            model_type: Type of model ("pytorch", "onnx", etc.)
        """
        self.model = model
        self.model_name = "Emotion"
        self.model_type = model_type

    def _run_inference(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded model
        Args:
            img_batch: Batch of preprocessed images (N, H, W, C) - for emotion: (N, 48, 48, 1)
        Returns:
            Emotion predictions as numpy array (N, 7) for 7 emotion classes
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
            
            # Convert to PyTorch tensor and adjust dimensions for PyTorch (N, C, H, W)
            # Input is (N, H, W, C), need to transpose to (N, C, H, W)
            tensor_input = torch.from_numpy(img_batch.transpose(0, 3, 1, 2)).float()
            
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
            
            # ONNX models typically expect (N, C, H, W) format
            # Convert from (N, H, W, C) to (N, C, H, W)
            img_batch_transposed = img_batch.transpose(0, 3, 1, 2)
            
            # Run inference
            outputs = self.model.run(None, {input_name: img_batch_transposed.astype(np.float32)})
            return outputs[0]
            
        except AttributeError:
            raise ValueError("Model must be an ONNX InferenceSession for onnx model type")

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess single image for emotion detection
        Args:
            img: Input image (224, 224, 3)
        Returns:
            Preprocessed grayscale image (48, 48)
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        return img_gray

    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Predict emotion probabilities for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n, n_emotions)
            where n_emotions is the number of emotion categories
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        processed_imgs = np.expand_dims(np.array([self._preprocess_image(img) for img in imgs]), axis=-1)

        # Prediction
        predictions = self._predict_internal(processed_imgs)

        return predictions


def create_pytorch_emotion_model(num_classes: int = 7) -> Any:
    """
    Create a PyTorch emotion model architecture
    Args:
        num_classes: Number of emotion classes (default 7)
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class EmotionModel(nn.Module):
            def __init__(self, num_classes=7):
                super(EmotionModel, self).__init__()
                
                # 1st convolution layer
                self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=0)
                self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)
                
                # 2nd convolution layer
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
                self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2)
                
                # 3rd convolution layer
                self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
                self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
                self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=2)
                
                # Calculate the size after convolutions for the first linear layer
                # This would need to be calculated based on the exact architecture
                # For now using a placeholder - you should calculate this properly
                self.fc1 = nn.Linear(128 * 2 * 2, 1024)  # Adjust size as needed
                self.dropout1 = nn.Dropout(0.2)
                self.fc2 = nn.Linear(1024, 1024)
                self.dropout2 = nn.Dropout(0.2)
                self.fc3 = nn.Linear(1024, num_classes)
                
            def forward(self, x):
                # 1st conv block
                x = F.relu(self.conv1(x))
                x = self.pool1(x)
                
                # 2nd conv block
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.avgpool1(x)
                
                # 3rd conv block
                x = F.relu(self.conv4(x))
                x = F.relu(self.conv5(x))
                x = self.avgpool2(x)
                
                # Flatten and fully connected layers
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout1(x)
                x = F.relu(self.fc2(x))
                x = self.dropout2(x)
                x = self.fc3(x)
                x = F.softmax(x, dim=1)
                
                return x
        
        return EmotionModel(num_classes)
        
    except ImportError:
        raise ImportError("PyTorch is required to create PyTorch emotion model")


def load_pytorch_emotion_model(model_path: Union[str, None] = None) -> Any:
    """
    Load a pre-trained PyTorch emotion model
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
            model = create_pytorch_emotion_model()
            logger.info("Created new PyTorch emotion model - weights need to be loaded separately")
            
        model.eval()
        return model
        
    except ImportError:
        raise ImportError("PyTorch is required to load PyTorch emotion model")


def load_onnx_emotion_model(model_path: str) -> Any:
    """
    Load an ONNX emotion model
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


def convert_keras_weights_to_pytorch(h5_weights_path: str, pytorch_model: Any) -> Any:
    """
    Helper function to convert Keras weights to PyTorch format
    Args:
        h5_weights_path: Path to the Keras .h5 weights file
        pytorch_model: PyTorch model to load weights into
    Returns:
        PyTorch model with loaded weights
    """
    try:
        import torch
        import h5py
        
        logger.info(f"Converting Keras weights from {h5_weights_path} to PyTorch format")
        
        # This is a placeholder - actual implementation would need to:
        # 1. Load the .h5 file
        # 2. Map Keras layer names to PyTorch layer names
        # 3. Convert weight formats (e.g., Conv2D weights need transposition)
        # 4. Load weights into the PyTorch model
        
        logger.info("Weight conversion not implemented - you need to implement the conversion logic")
        
        return pytorch_model
        
    except ImportError as e:
        raise ImportError(f"Required libraries not available: {e}")
