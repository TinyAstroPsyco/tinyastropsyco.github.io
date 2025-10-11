/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:18,March, 2024
   DESCRIPTION:The code demonstrates corner detection using the Harris corner detection algorithm. It captures video frames from the camera,
   converts them to grayscale, and applies the corner detection algorithm. One can adjust the threshold value interactively through a trackbar.
   Detected corners are visualized in real-time by drawing circles on the image
*/


#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat src_gray;// Stores the grayscale version of the input frame
int thresh = 200;// Threshold value for corner detection, we initially set it to 200 
int max_thresh = 255;//This is the maximum threshold value 
const char* source_window = "Source image"; // Window name for the source image 
const char* corners_window = "Corners detected";  // Window name for the detected corners


/*
   Description: Function to demonstrate corner detection using the Harris corner detection algorithm.
   Input: Two integer arguments, but we do not necessarily need to use them.
   Output: Displays detected corners on the image window.

    The function defintion is written below after the main function
*/
void cornerHarris_func(int, void*);

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    // This is a window to display the source image
    namedWindow(source_window);

    // Create a trackbar to adjust the threshold interactively
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_func);

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Frame is empty!" << endl;
            break;
        }
        //We are converting the frame into grayscale
        cvtColor(frame, src_gray, COLOR_BGR2GRAY);

        // This is where we display the frame
        imshow(source_window, frame);

        //Calling the  cornerHarris function 
        cornerHarris_func(0, 0);

        // Wait for key press
        char c = (char)waitKey(25);
        if (c == 27) // press "esc" key to exit the loop 
            break;
    }

    cap.release(); //Release the video capture object
    destroyAllWindows(); // Close all windows

    return 0;
}


//Function defintion for the cornerHarris_func
void cornerHarris_func(int, void*) {
    if (src_gray.empty()) {
        cout << "Source image (grayscale) is empty!" << endl;
        return;
    }


    // Set parameters for the Harris corner detection algorithm
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros(src_gray.size(), CV_32FC1);

    // Apply the Harris corner detection algorithm
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
    Mat dst_norm, dst_norm_scaled;
    //Normalising the values, so that it lies between 0 and 255
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    // Draw circles at detected corner locations
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int)dst_norm.at<float>(i, j) > thresh) {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    // Display the result in a window
    namedWindow(corners_window);
    imshow(corners_window, dst_norm_scaled);
}
