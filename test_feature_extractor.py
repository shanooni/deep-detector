import unittest
import torch
import torch.nn as nn
from torchvision.models import resnet152
from feature_extractor import load_resnet_model


class TestLoadResnetModel(unittest.TestCase):
    """Test cases for the load_resnet_model function."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_loads_resnet152_model(self):
        """Test that load_resnet_model correctly loads the ResNet152 model."""
        model = load_resnet_model(self.device)
        
        # Load a reference ResNet152 model to compare architecture
        reference_model = resnet152(weights=None, progress=False)
        reference_layers = list(reference_model.children())[:-1]
        
        # Check that the model is a Sequential container
        self.assertIsInstance(model, nn.Sequential)
        
        # Check that the number of layers matches (all ResNet152 layers except the last one)
        self.assertEqual(len(model), len(reference_layers))
        
        # Verify the model is in eval mode
        self.assertFalse(model.training)
        
        # Check that the model is on the correct device
        self.assertEqual(next(model.parameters()).device.type, self.device.type)

    def test_removes_final_classification_layer(self):
        """Test that load_resnet_model correctly removes the final classification layer."""
        model = load_resnet_model(self.device)
        
        # Load a full ResNet152 model for comparison
        full_resnet = resnet152(weights=None, progress=False)
        
        # The full ResNet152 has a final FC layer (Linear layer)
        # Check that the last layer of the full model is a Linear layer
        self.assertIsInstance(list(full_resnet.children())[-1], nn.Linear)
        
        # Check that our model does NOT end with a Linear layer
        # Instead, it should end with AdaptiveAvgPool2d
        last_layer = list(model.children())[-1]
        self.assertIsInstance(last_layer, nn.AdaptiveAvgPool2d)
        self.assertNotIsInstance(last_layer, nn.Linear)
        
        # Verify the model has one fewer layer than the full ResNet152
        self.assertEqual(len(list(model.children())), len(list(full_resnet.children())) - 1)
        
        # Test that the output shape is correct (should be feature vector, not class probabilities)
        # ResNet152 outputs 2048-dimensional features before the FC layer
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            output = model(dummy_input)
            # Output should be (batch_size, 2048, 1, 1)
            self.assertEqual(output.shape, (1, 2048, 1, 1))


if __name__ == "__main__":
    unittest.main()
