# built-in dependencies
from typing import Tuple

# 3rd party
import numpy as np
import cv2
import torch
from torchvision import transforms


def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """Normalize input image.

    Args:
        img (numpy array): the input image.
        normalization (str, optional): the normalization technique. Defaults to "base",
        for no normalization.

    Returns:
        numpy array: the normalized image.
    """

    # issue 131 declares that some normalization techniques improves the accuracy

    if normalization == "base":
        return img

    # @trevorgribble and @davedgd contributed this feature
    # restore input in scale of [0, 255] because it was normalized in scale of
    # [0, 1] in preprocess_face
    img *= 255

    if normalization == "raw":
        pass  # return just restored pixels

    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif normalization == "Facenet2018":
        # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
        img /= 127.5
        img -= 1

    elif normalization == "VGGFace":
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif normalization == "VGGFace2":
        # mean subtraction based on VGGFace2 training data
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif normalization == "ArcFace":
        # Reference study: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")

    return img


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to expected size of a ml model with adding black pixels.
    
    LEGACY FUNCTION: For PyTorch/ONNX models, consider using resize_image_pytorch() instead.
    
    Args:
        img (np.ndarray): pre-loaded image as numpy array
        target_size (tuple): input shape of ml model
    Returns:
        img (np.ndarray): resized input image
    """
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    # Put the base image in the middle of the padded image
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # make it 4-dimensional how ML models expect (batch_size, height, width, channels)
    img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img


def numpy_to_torch(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    Args:
        img (np.ndarray): Input image as numpy array with shape (batch, height, width, channels)
        device (str): Target device for the tensor ('cpu' or 'cuda')

    Returns:
        torch.Tensor: PyTorch tensor with shape (batch, channels, height, width)
    """
    # Convert from (batch, height, width, channels) to (batch, channels, height, width)
    if len(img.shape) == 4:
        img = np.transpose(img, (0, 3, 1, 2))
    elif len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

    tensor = torch.from_numpy(img).float()
    return tensor.to(device)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.

    Args:
        tensor (torch.Tensor): Input tensor with shape (batch, channels, height, width)

    Returns:
        np.ndarray: Numpy array with shape (batch, height, width, channels)
    """
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to numpy
    img = tensor.detach().numpy()

    # Convert from (batch, channels, height, width) to (batch, height, width, channels)
    if len(img.shape) == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    elif len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))

    return img


def get_pytorch_transforms(target_size: Tuple[int, int], normalization: str = "base"):
    """
    Get PyTorch transforms for image preprocessing.

    Args:
        target_size (Tuple[int, int]): Target image size (height, width)
        normalization (str): Normalization type

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]

    if normalization == "Facenet":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    elif normalization == "Facenet2018":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    elif normalization == "VGGFace":
        # Convert to [0, 255] range for VGGFace normalization
        transform_list.append(transforms.Lambda(lambda x: x * 255))
        transform_list.append(transforms.Normalize(mean=[93.5940/255, 104.7624/255, 129.1863/255],
                                                 std=[1.0, 1.0, 1.0]))
    elif normalization == "VGGFace2":
        transform_list.append(transforms.Lambda(lambda x: x * 255))
        transform_list.append(transforms.Normalize(mean=[91.4953/255, 103.8827/255, 131.0912/255],
                                                 std=[1.0, 1.0, 1.0]))
    elif normalization == "ArcFace":
        transform_list.append(transforms.Normalize(mean=[127.5/255, 127.5/255, 127.5/255],
                                                 std=[128/255, 128/255, 128/255]))

    return transforms.Compose(transform_list)


def resize_image_pytorch(img: np.ndarray, target_size: Tuple[int, int]):
    """
    Resize an image to expected size using PyTorch transforms.
    
    Args:
        img (np.ndarray): pre-loaded image as numpy array
        target_size (tuple): input shape of ml model (height, width)
    
    Returns:
        torch.Tensor: resized input image as PyTorch tensor with shape (1, channels, height, width)
    """
    # Ensure image is in uint8 format for transforms
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])    # Apply transform and add batch dimension
    tensor = transform(img)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.unsqueeze(0)
    
    return tensor
