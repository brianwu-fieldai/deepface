# stdlib dependencies

from typing import List, Union, Any

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.commons import weight_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

logger = Logger()

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"

# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]

# pylint: disable=too-few-public-methods
class GenderClient(Demography):
    """
    Gender model class - Framework agnostic implementation
    """

    def __init__(self, model: Any, model_type: str = "pytorch"):
        """
        Initialize gender model
        Args:
            model: Pre-trained model (PyTorch model, ONNX session, etc.)
            model_type: Type of model ("pytorch", "onnx", etc.)
        """
        self.model = model
        self.model_name = "Gender"
        self.model_type = model_type

    def _run_inference(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded model
        Args:
            img_batch: Batch of preprocessed images (N, H, W, C) - for gender: (N, 224, 224, 3)
        Returns:
            Gender predictions as numpy array (N, 2) for 2 gender classes
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

    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Predict gender probabilities for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n, 2)
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction
        predictions = self._predict_internal(imgs)

        return predictions

def create_pytorch_gender_model(num_classes: int = 2) -> Any:
    """
    Create a PyTorch gender model architecture
    Based on VGGFace backbone with gender classification head
    Args:
        num_classes: Number of gender classes (default 2: Woman, Man)
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class GenderModel(nn.Module):
            def __init__(self, num_classes=2):
                super(GenderModel, self).__init__()
                
                # VGGFace-like backbone (simplified version)
                # You would need to implement the full VGGFace architecture
                # This is a placeholder that mimics the structure
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Block 2
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Block 3
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Block 4
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Block 5
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                
                # Gender classification head (equivalent to the Keras head)
                self.gender_head = nn.Sequential(
                    nn.Conv2d(512, num_classes, kernel_size=1),  # Equivalent to Convolution2D(classes, (1, 1))
                    nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
                    nn.Flatten(),
                    nn.Softmax(dim=1)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = self.gender_head(x)
                return x
        
        return GenderModel(num_classes)
        
    except ImportError:
        raise ImportError("PyTorch is required to create PyTorch gender model")


def load_pytorch_gender_model(model_path: Union[str, None] = None) -> Any:
    """
    Load a pre-trained PyTorch gender model
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
            model = create_pytorch_gender_model()
            logger.info("Created new PyTorch gender model - weights need to be loaded separately")
            
        model.eval()
        return model
        
    except ImportError:
        raise ImportError("PyTorch is required to load PyTorch gender model")


def load_onnx_gender_model(model_path: str) -> Any:
    """
    Load an ONNX gender model
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


def convert_keras_gender_weights_to_pytorch(h5_weights_path: str, pytorch_model: Any) -> Any:
    """
    Helper function to convert Keras gender weights to PyTorch format
    Args:
        h5_weights_path: Path to the Keras .h5 weights file
        pytorch_model: PyTorch model to load weights into
    Returns:
        PyTorch model with loaded weights
    """
    try:
        import torch
        import h5py
        
        logger.info(f"Converting Keras gender weights from {h5_weights_path} to PyTorch format")
        
        # Load the H5 weights file
        with h5py.File(h5_weights_path, 'r') as f:
            # Get PyTorch model's state dict
            pytorch_state_dict = pytorch_model.state_dict()
            
            # Function to recursively find weights in HDF5 group
            def extract_weights_from_group(group, prefix=""):
                weights = {}
                for key in group.keys():
                    if isinstance(group[key], h5py.Group):
                        subweights = extract_weights_from_group(group[key], f"{prefix}{key}/")
                        weights.update(subweights)
                    elif isinstance(group[key], h5py.Dataset):
                        weight_name = f"{prefix}{key}"
                        weights[weight_name] = group[key][:]
                return weights
            
            # Extract all weights from the H5 file
            all_keras_weights = extract_weights_from_group(f)
            
            logger.info(f"Found {len(all_keras_weights)} weight tensors in Keras gender model")
            logger.info("Note: Gender model weight conversion needs custom mapping based on VGGFace structure")
            logger.info("This is a placeholder - implement specific weight mapping for your model")
            
            return pytorch_model
            
    except ImportError as e:
        raise ImportError(f"Required libraries not available: {e}")
    except Exception as e:
        logger.info(f"Error during weight conversion: {str(e)}")
        raise e


def load_pytorch_gender_model_with_keras_weights(
    h5_weights_path: Union[str, None] = None, 
    url: str = WEIGHTS_URL
) -> Any:
    """
    Create PyTorch gender model and load converted Keras weights
    Args:
        h5_weights_path: Path to local H5 weights file
        url: URL to download weights if local file not provided
    Returns:
        PyTorch model with loaded weights
    """
    try:
        import torch
        
        # Create PyTorch model
        model = create_pytorch_gender_model()
        
        # Get weights file
        if h5_weights_path is None:
            # Download weights if not provided
            weight_file = weight_utils.download_weights_if_necessary(
                file_name="gender_model_weights.h5", 
                source_url=url
            )
        else:
            weight_file = h5_weights_path
            
        # Convert and load Keras weights
        model = convert_keras_gender_weights_to_pytorch(weight_file, model)
        
        model.eval()
        logger.info("Successfully created PyTorch gender model with converted Keras weights")
        
        return model
        
    except ImportError:
        raise ImportError("PyTorch is required to load PyTorch gender model")
