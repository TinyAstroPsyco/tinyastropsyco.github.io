'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: Task 2 : Examine the network A & B: Analyze the first layer and show the effects of the filters.
'''
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
import cv2  


# Building the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
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


def apply_filters_and_plot(image, weights):
    """
    Apply the given filters one by one to the image and plot the results.
    Args:
    - image (torch.Tensor): Input image tensor.
    - weights (torch.Tensor): Filter weights tensor.
    """
    fig, axes = plt.subplots(5, 4, figsize=(12, 15)) # Setting up the plot shape and dimensions
    
    for i, ax in enumerate(axes.flat):
        if i < 20:
            if i % 2 == 0:  # Plot filter
                filter_index = i // 2
                filter = weights[filter_index, 0].numpy()
                ax.imshow(filter, cmap='gray')
                ax.set_title(f'Filter {filter_index}')

            else:  # Plot filtered image
                image_index = i // 2
                filter = weights[image_index, 0].numpy()  # Corrected index here
                with torch.no_grad():  # Disable gradient calculation
                    filtered_image = cv2.filter2D(image.numpy(), -1, filter)
                ax.imshow(filtered_image, cmap='gray')
                ax.set_title(f'Filtered Image {image_index}')
            ax.axis('off')
    plt.savefig('./filter_plot_with_filters.png')
    plt.show()


def read_model(model_path):
    """
    Load a pre-trained model from the given model_path.
    Parameters:
        model_path (str): Path to the saved model file.
    Returns:
        model (Net): Loaded pre-trained model.
    """
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def filter_plot(weights):
    """
    Plot and save the filters of a convolutional layer using pyplot.
    Parameters:
        weights (Tensor): The weights of the convolutional layer.
    """
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:  
            ax.imshow(weights[i, 0], cmap='viridis')
            ax.set_title(f'Filter {i}')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off') 
    plt.savefig('./filter_plot1.png') 
    plt.show()



# Main function
def main():
    # Reading the model and printing it
    print(f'Reading the model... .... ....')
    model_path = '.model.pth'  
    model = read_model(model_path)
    print(f'printing the model')
    print(model)

    # Checking the weights of the layer 1 from the loaded model and printing the weights
    weights = model.conv1.weight.data
    print("weights' shape:", weights.shape)
    print("filter weights of 1st layer:", weights[0, 0])
    
    
    # Visualizing the 10 filters
    filter_plot(weights)
    plt.pause(1)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Applying the 10 filters on the first training example image
    first_image, _ = train_dataset[0]
    plt.imshow(first_image.squeeze(), cmap= 'gray')
    weights = model.conv1.weight.detach()
    print(f'The weights shape is {weights.shape}')
    print(f'The first filter is :\n {weights[0,0]}')
    apply_filters_and_plot(first_image[0], weights) 
    plt.pause(1)

if __name__ == '__main__':
    main()
