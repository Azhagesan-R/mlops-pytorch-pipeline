import torch
import pytest
from src.model import SimpleCNN

def test_model_forward_shape():
    """Verify that the model processes a batch of images and outputs correct shapes."""
    batch_size = 4
    num_classes = 10
    model = SimpleCNN(num_classes=num_classes)
    
    # Standard format: [Batch Size, Channels, Height, Width]
    sample_input = torch.rand(batch_size, 3, 32, 32)
    output = model(sample_input)
    
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"

def test_model_parameter_gradients():
    """Ensure that backpropagation runs cleanly through the neural network layers."""
    model = SimpleCNN()
    sample_input = torch.rand(2, 3, 32, 32)
    output = model(sample_input)
    loss = output.sum()
    loss.backward()
    
    # Confirm weight parameters are collecting gradients correctly
    assert model.conv1.weight.grad is not None, "Gradients failed to propagate to conv1 weights"
