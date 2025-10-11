'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: Extension 3 : Detect the digits shown in the video stream using the pretrained model from task1.
'''

# Import necessary libraries
import torch
import cv2
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F

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


# Define a function to preprocess the frame
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Convert frame to grayscale
    resized_frame = cv2.resize(gray_frame, (28, 28))# Resize frame to 28x28 pixels
    inverted_frame = cv2.bitwise_not(resized_frame)# Invert colors
    tensor_frame = ToTensor()(inverted_frame).unsqueeze(0)# Convert frame to tensor and add batch dimension
    return tensor_frame
    
# Define a function to perform digit recognition
def recognize_digit(frame, device, model):
    with torch.no_grad():
        output = model(frame.to(device))  # Move frame to device before passing it to the model
        prediction = torch.argmax(F.softmax(output, dim=1)).item()
    return prediction

def main():
    device = torch.device('cpu')
    # Load the trained model for the file path
    model = Net()
    model.load_state_dict(torch.load(".model.pth", map_location=torch.device('cpu')))
    model.eval()     
    cap = cv2.VideoCapture(0) # Begin the video stream from the webcam on the computer, 0 is the default camera and 1 for the irium cam
    window_size = 5
    prediction_window = []
    
    while True:
        # Capture the frame
        _, frame = cap.read()        
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        print("Frame preprocessed.")
        # Recognise the digit
        digit_prediction = recognize_digit(processed_frame, device, model)
        print(f'Recognised : {digit_prediction}')
        # Update prediction window
        if len(prediction_window) < window_size:
            prediction_window.append(digit_prediction)
        else:
            prediction_window.pop(0)
            prediction_window.append(digit_prediction)
        # Finding the most predicted the digits
        final_prediction = max(set(prediction_window), key=prediction_window.count)
        # Displaying the recognised number
        cv2.putText(frame, str(final_prediction), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Live Digit Recognition', frame) #Showing the image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()