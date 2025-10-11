'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: Task 1 E & F: Read the network and run it on the test set and test the network on new inputs.
'''


# 
# Importing the required libraries
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image,  ImageOps, ImageEnhance

# Building the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() # Default is 50%
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


def load_model(model_path):
    """
    Load a pre-trained neural network model from the given path.
    Args:
    - model_path (str): Path to the saved model file.
    Returns:
    - model: Loaded neural network model.
    """
    model = Net()
    model.load_state_dict(torch.load(model_path))
    # Setting the network to eval mode
    model.eval()
    return model

# Function to plot the images
def plot_images(images, labels, preds):
    """
    Plot a grid of images along with their true and predicted labels.
    Args:
    - images (numpy.ndarray): Array of images to be plotted.
    - labels (numpy.ndarray): Array of true labels corresponding to the images.
    - preds (numpy.ndarray): Array of predicted labels corresponding to the images.
    """
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap = 'gray')
        # ax.imshow(images[i].reshape(28, 28), cmap='binary')
        ax.set_title(f'Pred: {preds[i]}, True: {labels[i]}')
        ax.axis('off')
    plt.show()


# Function to process the custom images 
def process_custom_image(image_path):
    """
    Process a custom image for input to a neural network model.
    Args:
    - image_path (str): Path to the custom image.
    Returns:
    - img_tensor (torch.Tensor): Processed image tensor ready for model input.
    """
    img = Image.open(image_path).convert('L') # Converting the image to grayscale
    img = img.resize((28, 28))
    # img = ImageOps.invert(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0) 
    img_array = np.array(img) / 255.0
    img_array = 1.0 - img_array
    img_array = (img_array - 0.1307) / 0.3081
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).float()
    plt.plot(img_array)
    return img_tensor


# Function to classify and plot the images
def classify_and_plot(model, image_tensor):
    """
    Classify a given image tensor using the provided neural network model and plot the image.
    Args:
    - model (torch.nn.Module): Neural network model for classification.
    - image_tensor (torch.Tensor): Processed image tensor ready for model input.
    """
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    plt.imshow(image_tensor.squeeze().numpy(), cmap='gray')
    plt.title(f'Predicted: {pred.item()}')
    plt.axis('off')
    plt.show()


# 
def process_and_classify_custom_images(model, image_paths):
    """
    Process and classify a list of custom images using the provided neural network model.
    Args:
    - model (torch.nn.Module): Neural network model for classification.
    - image_paths (list): List of paths to custom images.
    """
    for image_path in image_paths:
        img_tensor = process_custom_image(image_path)
        classify_and_plot(model, img_tensor)


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])    
    # transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])    

    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = load_model('.model.pth') # Loading the model
    print(model)

# Test the trained model on the test set in MNIST dataset and visualize the first 9 digits
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].squeeze().numpy()
            correct = target.numpy()
            output_values = np.round(output.numpy(), 2)
            
            for i in range(10):
                print(f'Output Values: {output_values[i]} Max Index: {pred[i]} Correct Label: {correct[i]}')
            plot_images(data.numpy(), correct, pred)
            break  


# Test the trained model on new inputs
    custom_image_paths = ['./custom_image_data_set/0.png', './custom_image_data_set/1.png', './custom_image_data_set/2.png',
                          './custom_image_data_set/3.png', './custom_image_data_set/4.png', './custom_image_data_set/5.png',
                          './custom_image_data_set/6.png', './custom_image_data_set/7.png', './custom_image_data_set/8.png',
                          './custom_image_data_set/9.png']
    process_and_classify_custom_images(model, custom_image_paths)

if __name__ == '__main__':
    main()