'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: Task4: Design your own experiments
'''

# Importing the required libraries
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class FNetwork(nn.Module):
    def __init__(self, num_conv_layers, num_conv_filters, dropout_rates):
        """
        Initialize the FNetwork neural network model.
        Args:
        - num_conv_layers (int): Number of convolutional layers.
        - num_conv_filters (int): Number of convolutional filters.
        - dropout_rates (float): Dropout rate for dropout layers.
        """
        super(FNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_filter_size = num_conv_filters
        self.dropout_rate = dropout_rates

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=num_conv_filters, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Add additional convolutional layers based on num_conv_layers
        for _ in range(1, num_conv_layers):
            self.conv_layers.add_module("Conv2d", nn.Conv2d(16, 16, kernel_size=num_conv_filters, stride=1, padding=1))
            self.conv_layers.add_module("ReLU", nn.ReLU())
            self.conv_layers.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rates)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self._calculate_flattened_size(), 512),
            nn.ReLU(),
            nn.Dropout(dropout_rates),
            nn.Linear(512, 10)
        )

    def _calculate_flattened_size(self):
        """
        Calculate the size of the flattened feature map.
        Returns:
        - int: Size of the flattened feature map.
        """
        size = 28  
        padding = 1
        stride = 1
        for _ in range(self.num_conv_layers):
            size = (size - self.conv_filter_size + 2 * padding) / stride + 1 
            size = size / 2  

        final_size = int(size * size * 16)  
        return final_size

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
        - x (torch.Tensor): Input tensor.
        Returns:
        - torch.Tensor: Output tensor (logits).
        """
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_network(dataloader, model, loss_fn, optimizer, device):
    """
    Train the neural network model.
    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader for training data.
    - model (torch.nn.Module): Neural network model.
    - loss_fn: Loss function.
    - optimizer: Optimizer for updating model parameters.
    - device (torch.device): Device to perform training on (e.g., 'cpu' or 'cuda').
    Returns:
    - None
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute predictions and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print training progress
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_network(dataloader, model, loss_fn,device):
    """
    Test the neural network model.
    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader for test data.
    - model (torch.nn.Module): Neural network model.
    - loss_fn: Loss function.
    - device (torch.device): Device to perform testing on cpu
    Returns:
    - None
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main(argv):
    # Load FashionMNIST dataset and create DataLoader objects
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Choosing the device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Define hyperparameter options
    num_conv_layers = 2
    num_conv_filters = 3
    dropout_rate = 0.2
    num_conv_layers_options = [1, 2]
    conv_filter_sizes_options = [1, 2, 3, 4, 5]
    dropout_rates_options = [0.1, 0.2, 0.3, 0.4, 0.5]
    epochs = 5

    # Hyperparameter tuning loop
    for num_conv_layers in num_conv_layers_options:
        model = FNetwork(num_conv_layers, num_conv_filters, dropout_rate).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        print(f"Optimizing: num_conv_layers={num_conv_layers}, num_conv_filters={num_conv_filters}, dropout_rate={dropout_rate}")
        for t in range(epochs):
            train_network(train_dataloader, model, loss_fn, optimizer, device)
            test_network(test_dataloader, model, loss_fn, device)

    for num_conv_filters in conv_filter_sizes_options:
        num_conv_layers = 2
        dropout_rate = 0.1
        model = FNetwork(num_conv_layers, num_conv_filters, dropout_rate).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        print(f"Optimizing: num_conv_layers={num_conv_layers}, num_conv_filters={num_conv_filters}, dropout_rate={dropout_rate}")
        for t in range(epochs):
            print(f"Epoch {t+1}")
            train_network(train_dataloader, model, loss_fn, optimizer, device)
            test_network(test_dataloader, model, loss_fn, device)

    for num_conv_layers in num_conv_layers_options:
        num_conv_filters = 2
        dropout_rate = 0.1
        model = FNetwork(num_conv_layers, num_conv_filters, dropout_rate).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        print(f"Optimizing: num_conv_layers={num_conv_layers}, num_conv_filters={num_conv_filters}, dropout_rate={dropout_rate}")
        for t in range(epochs):
            print(f"Epoch {t+1}")
            train_network(train_dataloader, model, loss_fn, optimizer, device)
            test_network(test_dataloader, model, loss_fn, device)

    for dropout_rate in dropout_rates_options:
        num_conv_layers = 2
        num_conv_filters = 3
        model = FNetwork(num_conv_layers, num_conv_filters, dropout_rate).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        print(f"Optimizing: num_conv_layers={num_conv_layers}, num_conv_filters={num_conv_filters}, dropout_rate={dropout_rate}")
        for t in range(epochs):
            print(f"Epoch {t+1}")
            train_network(train_dataloader, model, loss_fn, optimizer, device)
            test_network(test_dataloader, model, loss_fn, device)

if __name__ == "__main__":
    main(sys.argv)
