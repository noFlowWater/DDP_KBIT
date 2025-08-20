"""
Data transformation functions for processing MNIST and other image data.

This module contains transformation functions for various data formats:
- transform_MNISTData: Transform flat MNIST image data to tensors
- transform_RawMNISTData: Transform raw image bytes to tensors
- transform_mongodb_image: Transform MongoDB image lists to tensors
"""

import torch
import numpy as np
from PIL import Image
import io
from torchvision import transforms


def transform_MNISTData(image_data):
    """
    Transform image data to tensor format.
    
    Args:
        image_data (list): Flat image data list.
    
    Returns:
        torch.Tensor: Transformed image tensor with shape [1, 28, 28].
    """
    image_flat = np.array(image_data)
    image_tensor = torch.tensor(image_flat, dtype=torch.float32).view(1, 28, 28)
    return image_tensor


def transform_RawMNISTData(image_bytes):
    """
    Transform RawMNISTData to tensor format.
    
    Args:
        image_bytes (bytes): Raw image bytes data.
    
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    # Define transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  # Uncomment for normalization
    ])
    
    # Convert image data from bytes to PIL image
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
    
    # Apply transformation pipeline
    image_tensor = transform_pipeline(image)
    
    return image_tensor


def transform_mongodb_image(image_list):
    """
    Transform MongoDB image list to tensor format.
    
    Args:
        image_list (list): Image data list (already normalized, 784 values).
    
    Returns:
        torch.Tensor: Transformed image tensor with shape [1, 28, 28].
    """
    # Convert list to numpy array and reshape appropriately
    image_array = np.array(image_list, dtype=np.float32)
    
    # Reshape 784 values to [1, 28, 28] format
    image_tensor = torch.tensor(image_array).view(1, 28, 28)
    
    return image_tensor


# MNISTImage Avro schema definition (same as used in mongodb_avro_experiment.ipynb)
mnist_avro_schema = {
    "type": "record",
    "name": "MNISTImage",
    "fields": [
        {"name": "data", "type": {"type": "array", "items": "float"}},
        {"name": "shape", "type": {"type": "array", "items": "int"}},
        {"name": "label", "type": "int"},
        {"name": "meta", "type": {
            "type": "record",
            "name": "ImageMetadata",
            "fields": [
                {"name": "dataset", "type": "string"},
                {"name": "split", "type": "string"},
                {"name": "index", "type": "int"}
            ]
        }}
    ]
}


def create_transform_pipeline(normalize=False, mean=(0.1307,), std=(0.3081,)):
    """
    Create a standard transformation pipeline for MNIST data.
    
    Args:
        normalize (bool): Whether to apply normalization
        mean (tuple): Mean values for normalization
        std (tuple): Standard deviation values for normalization
    
    Returns:
        torchvision.transforms.Compose: Transformation pipeline
    """
    transforms_list = [transforms.ToTensor()]
    
    if normalize:
        transforms_list.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(transforms_list)


def transform_image_from_pil(pil_image, normalize=False):
    """
    Transform PIL Image to tensor format with optional normalization.
    
    Args:
        pil_image (PIL.Image): PIL Image object
        normalize (bool): Whether to apply MNIST normalization
    
    Returns:
        torch.Tensor: Transformed image tensor
    """
    transform_pipeline = create_transform_pipeline(normalize=normalize)
    
    # Ensure image is in grayscale mode
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    return transform_pipeline(pil_image)


def transform_numpy_to_tensor(numpy_array, target_shape=(1, 28, 28)):
    """
    Transform numpy array to tensor with specified shape.
    
    Args:
        numpy_array (np.ndarray): Input numpy array
        target_shape (tuple): Target tensor shape
    
    Returns:
        torch.Tensor: Reshaped tensor
    """
    if not isinstance(numpy_array, np.ndarray):
        numpy_array = np.array(numpy_array, dtype=np.float32)
    
    tensor = torch.tensor(numpy_array, dtype=torch.float32)
    
    # Reshape if necessary
    if tensor.numel() == np.prod(target_shape):
        tensor = tensor.view(target_shape)
    
    return tensor


def batch_transform_images(image_batch, transform_fn):
    """
    Apply transformation function to a batch of images.
    
    Args:
        image_batch (list): List of images to transform
        transform_fn (callable): Transformation function to apply
    
    Returns:
        list: List of transformed images
    """
    return [transform_fn(image) for image in image_batch]


def validate_tensor_shape(tensor, expected_shape):
    """
    Validate that tensor has expected shape.
    
    Args:
        tensor (torch.Tensor): Tensor to validate
        expected_shape (tuple): Expected shape
    
    Returns:
        bool: True if shape matches, False otherwise
    
    Raises:
        ValueError: If tensor shape doesn't match expected shape
    """
    if tensor.shape != expected_shape:
        raise ValueError(f"Expected tensor shape {expected_shape}, got {tensor.shape}")
    return True