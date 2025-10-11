
'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: Task 1 A & B & C & D: Build a network model and train it and save it to a file
'''


# 

# Importing the necessary libraries
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchviz import make_dot
import os


# Building the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() # Default dropout rate is 50%
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




def train(epoch, network, train_loader, optimizer, train_losses, train_counter):
    """
    Trains the neural network for one epoch.

    Args:
        epoch (int): Current epoch number.
        network (nn.Module): Neural network model.
        train_loader (DataLoader): DataLoader for training dataset.
        optimizer (Optimizer): Optimizer for updating model parameters.
        train_losses (list): List to store training losses.
        train_counter (list): List to store number of training examples seen.

    Returns:
        None
    """
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())


            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

            # Saving the network
            torch.save(network.state_dict(), '.model.pth')
            torch.save(optimizer.state_dict(), '.optimizer.pth')
            # print(f'Saved the network')


def test(network, test_loader, test_losses, test_accuracy):
    """
    Evaluates the neural network on the test dataset.

    Args:
        network (nn.Module): Neural network model.
        test_loader (DataLoader): DataLoader for test dataset.
        test_losses (list): List to store test losses.

    Returns:
        None
    """
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    
    
    test_loss /= len(test_loader.dataset)
    if test_losses is not None:
        test_losses.append(test_loss)
    accuracy = 100. * correct / total
    test_accuracy.append(accuracy)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')



def evaluation_plot(train_counter, train_losses, test_counter, test_losses):
    """
    Plots the training and test error over the course of training.

    Args:
        train_counter (list): List containing the number of training examples seen.
        train_losses (list): List containing the training losses.
        test_counter (list): List containing the number of test examples seen.
        test_losses (list): List containing the test losses.

    Returns:
        None
    """
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='salmon')
    plt.scatter(test_counter, test_losses, color='green')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()




def accuracy_plot(train_accuracies, test_accuracies):
    """
    Plots the training and test accuracies over epochs.

    Args:
        train_accuracies (list): List containing the training accuracies.
        test_accuracies (list): List containing the test accuracies.

    Returns:
        None
    """
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, test_accuracies[1:], label='Test Accuracy', color='green')
    plt.title('Training and Test Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()



# Main function
def main():
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    # Loading the training dataset
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size_train, shuffle=True)
        
    # Loading the testing dataset
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size_test, shuffle=False)
    
        
    # Displaying the first 6 images in the test dataset
    fig, axes = plt.subplots(1, 6, figsize=(15, 3)) # Create a figure with 1 row and 6 columns for subplots
    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= 6:  # Limit to the first 6 batches
            break
        for i in range(len(images)):
            image, label = images[i], labels[i] # Get the image and label from the batch
            print(f'shape : {image.shape}')
            subplot_idx = batch_idx * len(images) + i
            if subplot_idx < 6:  # Check if the subplot index is within bounds
                axes[subplot_idx].imshow(image.squeeze(), cmap='gray') # Display the image, using the squeeze method to remove the additional dimension added to the data
                axes[subplot_idx].set_title(f"Label: {label}") # Set the title with the corresponding label
                axes[subplot_idx].axis('off') # Turn off the axis

    plt.pause(0.1)  # Pause to ensure the figure is properly displayed
    plt.show() # Display the plot

    
    # Defining the varibles
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    train_accuracy = []
    test_accuracy = []


    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    
    # Visualizing the network
    dummy_input = torch.randn(1, 1, 28, 28)  # Create a dummy input tensor
    dot = make_dot(network(dummy_input), params=dict(network.named_parameters()))
    dot.format = 'png'  # Set the format to PNG (optional)
    dot.render('model', directory='./Results')  # Save the visualization as an image file

    # Training the model and evaluating the model on both the training and test sets after each epoch
    test(network, test_loader, test_losses, test_accuracy)  
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, optimizer, train_losses, train_counter)
        test(network, test_loader, test_losses, test_accuracy)
        # Calculating the accuracies
        test(network, train_loader, None, train_accuracy)
    print(f'The lenght of train loss = {len(train_losses)} and the lenght of the test losses = {len(test_losses)}')
    evaluation_plot(train_counter, train_losses, test_counter, test_losses)
    accuracy_plot(train_accuracy, test_accuracy)


if __name__ == "__main__":

    # Defining the hyperparameters
    n_epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1

    main()