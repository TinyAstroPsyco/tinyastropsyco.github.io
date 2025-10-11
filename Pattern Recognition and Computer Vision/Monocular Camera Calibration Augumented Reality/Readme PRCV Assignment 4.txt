Name: Poojit Maddineni
Names of group members: Venkata Satya Naga Sai Karthik Koduru

Links/URL's: None.

System Info:
Poojit Maddineni's Computer: Windows 10 Home - Operating System | IDE - Visual Studio.
Venkata Satya Naga Sai Karthik Koduru's Computer:Windows 11 - Operating System | IDE - Visual Studio Code.

Travel Time:

Venkata Satya Naga Sai Karthik Koduru : 2-Days
Poojit Maddineni : 2-Days


How to run the program:

You have to change the following file paths in the main program:

1. Tasks 1-3:
 - String calib_parameters = "D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\calibration_parameters.yml"; // Change the file location as per preference
 - imwrite("D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\image_" + to_string(image_number) + ".png", frame);//Save the correspondig image your own path

Use s to save image and c to calibrate.

2. Tasks 4-6:
 - string calib_file_name = "D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\calibration_parameters.yml"; // Change the directory here for the yml file

3. Extension 1:
 - string calib_file_name = "D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\calibration_parameters.yml";// Change the directory here for the yml file
 - Mat texture = imread("C:\\Users\\pooji\\Downloads\\Capitan_America_shield.jpeg"); //Give the file location of the mask image

4. Extension 3:
 - string calib_file_name = "D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\calibration_parameters.yml"; //Give the location of the camera calibration files
 - VideoCapture cap("C:\\Users\\pooji\\Downloads\\640.mp4");//Change the path here for using a different video
 - cv::VideoWriter video("<Specify the path to save your video to\\xxxxx.mp4>", cv::VideoWriter::fourcc('H', '2', '6', '4'), 30, cv::Size(frameWidth, frameHeight)); 

5. Extension 4:
 - This code needs the xfeatures2d contrib module to be run.
 - When user presses x it enters surf mode and if user presses y it enters sift mode
