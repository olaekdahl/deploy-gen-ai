"""
Simple PyTorch Linear Regression Example
Trains a small neural network on dummy data.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNet(nn.Module):
    """A minimal neural network for linear regression."""
    
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def create_dummy_data(n_samples=100):
    """Create dummy linear data: y = 2x + 1 + noise."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, 1)
    y = 2 * X + 1 + 0.1 * torch.randn(n_samples, 1)
    return X, y


def train(model, X, y, epochs=50, lr=0.1):
    """Train the model and print loss."""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    print(f"Training for {epochs} epochs...")
    print("-" * 30)
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:3d}/{epochs}] | Loss: {loss.item():.4f}")
    
    print("-" * 30)
    print("Training complete!")
    return model


def main():
    print("=" * 40)
    print("PyTorch Simple Neural Network Training")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU")
    print("=" * 40)
    
    # Create model and data
    model = SimpleNet()
    X, y = create_dummy_data()
    
    print(f"\nModel architecture:\n{model}\n")
    print(f"Training data shape: X={X.shape}, y={y.shape}\n")
    
    # Train
    trained_model = train(model, X, y)
    
    # Test prediction
    test_input = torch.tensor([[1.0]])
    with torch.no_grad():
        prediction = trained_model(test_input)
    print(f"\nTest: input=1.0 → prediction={prediction.item():.4f} (expected ~3.0)")


if __name__ == "__main__":
    main()
