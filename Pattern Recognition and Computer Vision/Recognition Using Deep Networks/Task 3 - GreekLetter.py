'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: task 2 part 3: Transfer Learning on Greek Letters.
'''

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
from PIL import Image
import math
from PIL import Image,  ImageOps, ImageEnhance

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


def load_model(model_path):
    """
    Load a pre-trained model from the specified path, freeze its parameters,
    and replace the last layer with a linear layer with three nodes.
    Args:
    - model_path (str): Path to the saved model file.
    Returns:
    - model: Loaded and modified neural network model.
    """
    model = Net()
    model.load_state_dict(torch.load(model_path))
    # Setting the model to evaluate mode
    model.eval()
    # Freezing the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(model.fc1.out_features, 3) # Replacing the last layer with a linear layer with three nodes  
    return model


def plot_training_error(training_errors):
    """
    Plot the training error over epochs.
    Args:
    - training_errors (list): List of training errors over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(training_errors, label='Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Error Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


# Processing the images
def preprocess_and_load_images(image_paths, transform):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image.unsqueeze(0))  
    return torch.cat(images, dim=0)  

# Classifying the images
def classify_images(model, images, class_names):
    """
    Preprocess and load a list of images using the specified transformation.
    Args:
    - image_paths (list): List of paths to image files.
    - transform (torchvision.transforms): Transformations to apply to the images.
    Returns:
    - torch.Tensor: Tensor containing preprocessed images.
    """
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for idx, pred in enumerate(predicted):
            print(f'Image {idx + 1}: Predicted class is {class_names[pred.item()]}')


# Custom transformation class for preprocessing Greek alphabet images.
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

def preprocess_and_load_images(image_paths):
    """
    Preprocess and load a list of images using a specific transformation pipeline.
    Args:
    - image_paths (list): List of paths to image files.
    Returns:
    - torch.Tensor: Tensor containing preprocessed images.
    """
    transform_pipeline = transforms.Compose([
        GreekTransform(),
        transforms.ToTensor(),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
       
        image = transform_pipeline(image)
        images.append(image.unsqueeze(0))  
    return torch.cat(images, dim=0)  

def process_custom_image(image_path):
    """
    Process a custom image for input to a neural network model.
    Args:
    - image_path (str): Path to the custom image.
    Returns:
    - img_tensor (torch.Tensor): Processed image tensor ready for model input.
    """
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img) / 255.0
    img_array = 1.0 - img_array
    img_array = (img_array - 0.1307) / 0.3081
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).float()
    return img_tensor


def classify_and_plot(model, image_tensor):
    
    """
    Classify a given image tensor using the provided neural network model and plot the image.
    Args:
    - model (torch.nn.Module): Neural network model for classification.
    - image_tensor (torch.Tensor): Processed image tensor ready for model input.
    """

    # Define a dictionary mapping class indices to Greek letter names
    greek_letter_names = {
        0: 'Alpha',
        1: 'Beta',
        2: 'Gamma'
    }
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True).item()
        predicted_letter = greek_letter_names[pred]  # Get the name corresponding to the predicted class index

    plt.imshow(image_tensor.squeeze().numpy(), cmap='binary')
    plt.title(f'Predicted: {predicted_letter}')  # Display the predicted letter name
    plt.axis('off')
    plt.show()

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
    training_set_path = './greek_train/greek_train'  
    greek_train = DataLoader(
        torchvision.datasets.ImageFolder(training_set_path, 
                                         transform=transforms.Compose([
                                             GreekTransform(),
                                             transforms.ToTensor(),
                                             transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ])),
        batch_size=5,
        shuffle=True
    )
    
    model = load_model('.model.pth')
    print(model)

    # Setting the hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.8)
    criterion = nn.CrossEntropyLoss()
    training_errors = []
    epochs = 2000
    perfect_threshold = 0.99  
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for data, target in greek_train:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(greek_train.dataset)
        accuracy = correct / len(greek_train.dataset)

        training_errors.append(avg_loss)
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        if accuracy >= perfect_threshold:
            print(f"Reached almost perfect accuracy after {epoch + 1} epochs.")
            break
    
    
    image_paths = [
        './handwritten_data/test1.png', './handwritten_data/test2.png', './handwritten_data/test3.png',
        './handwritten_data/test4.png', './handwritten_data/test5.png', './handwritten_data/test6.png',
        './handwritten_data/test7.png', './handwritten_data/test8.png', './handwritten_data/test9.png'
    ]
    # Processing and classifying the images
    process_and_classify_custom_images(model, image_paths)
    
if __name__ == '__main__':
    main()