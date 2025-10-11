'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: Extension 2
'''


# Importing the required libraries
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

def gabor_filter(kernel_size, sigma, theta, lambd, gamma, psi=0):
    """
    Function to generate a Gabor filter.
    Args:
    - kernel_size (int): Size of the filter kernel.
    - sigma (float): Standard deviation of the Gaussian envelope.
    - theta (float): Orientation of the normal to the parallel stripes of a Gabor function.
    - lambd (float): Wavelength of the sinusoidal factor.
    - gamma (float): Spatial aspect ratio.
    - psi (float, optional): Phase offset. Default is 0.
    Returns:
    - torch.Tensor: Gabor filter as a PyTorch tensor.
    """
    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax = ymax = kernel_size // 2
    xmin = ymin = -xmax
    (x, y) = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    gb *= np.cos(2 * np.pi * x_theta / lambd + psi)

    return torch.tensor(gb, dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gabor_filters = nn.Parameter(self.create_gabor_filters(), requires_grad=False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*5*5, 50)
        self.fc2 = nn.Linear(50, 10)

    def create_gabor_filters(self, num_filters=10):
        """
        Function to create Gabor filters.
        Args:
        - num_filters (int): Number of Gabor filters to create.
        Returns:
        - torch.Tensor: Gabor filters as a PyTorch tensor.
        """
        filters = []
        for i in range(num_filters):
            theta = np.pi * i / num_filters
            gabor = gabor_filter(kernel_size=5, sigma=1, theta=theta, lambd=3, psi=0, gamma=1)
            filters.append(gabor)
        filters = np.stack(filters, axis=0)
        filters = torch.from_numpy(filters).float()
        filters = filters.unsqueeze(1)
        return filters

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
        - x (torch.Tensor): Input tensor.
        Returns:
        - torch.Tensor: Output tensor.
        """
        x = F.conv2d(x, self.gabor_filters, padding=2)
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval):
    
    """
    Function to train the neural network.
    Args:
    - epoch (int): Current epoch number.
    - network (nn.Module): Neural network model.
    - train_loader (DataLoader): DataLoader for training data.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - train_losses (list): List to store training losses.
    - train_counter (list): List to store training step counters.
    - log_interval (int): Interval for logging training progress.
    """    
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            # saving the model
            torch.save(network.state_dict(), './model/model.pth')
            torch.save(optimizer.state_dict(), './model/optimizer.pth')

def test(network, test_loader, test_losses):
    """
    Function to evaluate the performance of the neural network on the test set.

    Args:
    - network (nn.Module): Neural network model.
    - test_loader (DataLoader): DataLoader for test data.
    - test_losses (list): List to store test losses.
    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    # Print test set performance
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    
def test_and_visualize(network, test_loader):
    """
    Function to evaluate the performance of the neural network on the test set and visualize some examples.
    Args:
    - network (nn.Module): Neural network model.
    - test_loader (DataLoader): DataLoader for test data.
    """
    network.eval()# Set the network to evaluation mode
    test_loss = 0
    correct = 0
    examples = [] # List to store example data for visualization
    
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data) # Forward pass
            test_loss += F.nll_loss(output, target, reduction='sum').item()# Calculate the loss
            pred = output.argmax(dim=1, keepdim=True) # Get the predicted labels
            correct += pred.eq(target.view_as(pred)).sum().item() # Count correct predictions
            
            # Collect example data for visualization
            if len(examples) < 9:# Limit to 9 examples
                for i in range(data.size(0)):
                    if len(examples) >= 9:
                        break
                    examples.append((data[i], target[i], pred[i]))
    
    test_loss /= len(test_loader.dataset) # Calculate average test loss
    # Print test set performance
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    visualize_examples(examples) # Visualize example data
    
def visualize_examples(examples):
    """
    Function to visualize example data along with their true and predicted labels.
    Args:
    - examples (list): List containing tuples of example data, true labels, and predicted labels.
    """
    fig = plt.figure(figsize=(9, 3)) # Create a figure to hold the subplots
    for i, (data, true_label, pred_label) in enumerate(examples):
        data, true_label, pred_label = data.cpu(), true_label.cpu(), pred_label.cpu()
        ax = fig.add_subplot(1, 9, i+1) # Add a subplot for each example
        ax.imshow(data.squeeze(), cmap='gray', interpolation='none') # Plot the example data
        # Set the title of the subplot with true and predicted labels
        ax.set_title(f'True: {true_label.item()}\nPred: {pred_label.item()}', fontsize=10)
        ax.set_xticks([]) # Remove x-axis ticks
        ax.set_yticks([]) # Remove y-axis ticks
    plt.show() # Display the figure with subplots

def main():
    # Defining the hyperparameters
    n_epochs = 5
    batch_size_train = 32
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1

    # Setting random seed for reproducibility
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False # Disabling cuDNN for deterministic results
    # Loading training and test datasets
    train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True,
                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Normalize((0.1307,), (0.3081,))])
                                             ), batch_size=batch_size_train, shuffle=True)

    test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                  transforms.Normalize((0.1307,), (0.3081,))])
                                            ), batch_size=batch_size_test, shuffle=False)

    train_losses = [] # List to store training losses
    train_counter = [] # List to store number of training examples seen during training
    test_losses = [] # List to store test losses
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # test(network, test_loader, test_losses)
    # for epoch in range(1, n_epochs + 1):
    #     train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval)
    #     test(network, test_loader, test_losses)

    # Initial test and visualization
    test_and_visualize(network, test_loader)
    # Training loop
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval)
        
        test_and_visualize(network, test_loader)
if __name__ == "__main__":
    main()

