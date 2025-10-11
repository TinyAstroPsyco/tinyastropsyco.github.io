Name: Poojit Maddineni
Names of group members: Venkata Satya Naga Sai Karthik Koduru

Links/URL's: None.

System Info:
Poojit Maddineni's Computer: Windows 10 Home - Operating System | IDE - Visual Studio.
Venkata Satya Naga Sai Karthik Koduru's Computer:Windows 11 - Operating System | IDE - Visual Studio Code.

Travel Time:

Venkata Satya Naga Sai Karthik Koduru : 1 Day
Poojit Maddineni : 1 Day


How to run the program:

You have to change the following file paths in the main program:



CSV File Path:
In the functions computeClosestLabel() and knn_matching(), there are references to a CSV file containing object features. The user needs to replace "D:\\MS Robotics Neu\\Spring 2024\\object_features.csv" with the actual path to their CSV file containing the object features.

Image File Paths:
In the section where images are saved with labels (under the 'n' key press event), file paths like "D:\\MS Robotics Neu\\Spring 2024\\binary_", "D:\\MS Robotics Neu\\Spring 2024\\dilated_", etc., are used. Users need to update these paths to the directory where they want to save their images.

Detected Image Paths:
Similar to the image saving part, paths like "D:\\MS Robotics Neu\\Spring 2024\\Detected_" are used to save detected images. Users should update these paths as per their directory structure and preferences.


Once you've adjusted the file path, execute the program and position the object in front of the camera. Press 'n' to save the feature vectors to the CSV file. Next, rerun the program and press 'd' for object detection; the detected object's label will display on the window. To generate a confusion matrix, place an object in front of the camera and press 'm'; you'll be prompted to input the label. Repeat this process for each object in various orientations; for instance, we repeated it 15 times (with 3 different orientations for 5 objects). This process generates a confusion matrix that compares true and predicted labels.
 