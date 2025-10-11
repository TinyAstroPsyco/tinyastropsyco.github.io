/*
   STUDENT NAME:Poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:18,March, 2024
   DESCRIPTION:The code captures video from the default camera, detecting keypoints using both SURF and SIFT algorithms. It utilizes OpenCV's video and feature detection modules to
   extract keypoints from each frame. SURF and SIFT detectors are applied separately, with the detected keypoints visualized in real-time. The code also handles potential errors such
   as failure to open the camera or capturing blank frames.
*/


// Including the necessary libraries
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp" 
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/nonfree/features2d.hpp" // For SIFT


//Name space declarations
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

//Main Code starts here
int main(int argc, char* argv[])
{
    VideoCapture cap(0); // 0 for default computer web cam and 1 for iruin webcam
    if (!cap.isOpened())  // Check if camera opened successfully
    {
        std::cout << "The camera failed to open, please check the camera connection!!" << std::endl;
        return -1;
    }

    Mat frame; //Initilizing the camera frame
    char algorithm;// Variable to store user input for algorithm selection
    while (1) //While true
    {
        // Capture the frame
        cap >> frame;
        if (frame.empty()) // Checking if the frame is empty
        {
            std::cout << "The frame is empty, please check the frame!!" << std::endl;
            break;
        }

        // Check for user input to select algorithm
        cout << "Press 'x' for SURF or 'y' for SIFT algorithm: ";
        cin >> algorithm;

        // Checking for key press, if ESC is pressed, code will break
        char key = (char)waitKey(25);
        if (key == 27) {
            break;
        }

        // Detect keypoints based on user selection
        if (algorithm == 'x' || algorithm == 'X') {
            //Detect the keypoints using SURF Detector
            int minHessian = 400; // Threshold to control the density of keypoints
            Ptr<SURF> surf_detector = SURF::create(minHessian);
            std::vector<KeyPoint> surf_keypoints;
            surf_detector->detect(frame, surf_keypoints);

            // Draw SURF keypoints
            Mat img_surf_keypoints;
            drawKeypoints(frame, surf_keypoints, img_surf_keypoints);
            imshow("SURF Keypoints", img_surf_keypoints);
        }
        else if (algorithm == 'y' || algorithm == 'Y') {
            // Detect the keypoints using SIFT Detector
            cv::SiftFeatureDetector sift_detector;
            std::vector<cv::KeyPoint> sift_keypoints;
            sift_detector.detect(frame, sift_keypoints);

            // Draw SIFT keypoints
            Mat img_sift_keypoints;
            cv::drawKeypoints(frame, sift_keypoints, img_sift_keypoints);

            // Show SIFT keypoints
            imshow("SIFT Keypoints", img_sift_keypoints);
        }
        else {
            cout << "Invalid input. Please press 'x' for SURF or 'y' for SIFT." << endl;
        }


        

        //Detect the keypoints using SURF Detector
        int minHessian = 400; //Thresholdig to control the density of the key points
        Ptr<SURF> surf_detector = SURF::create(minHessian);
        std::vector<KeyPoint> surf_keypoints;
        surf_detector->detect(frame, surf_keypoints); // Analysing the frame and detecting the key points.

        // Draw SURF keypoints
        Mat img_surf_keypoints;
        drawKeypoints(frame, surf_keypoints, img_surf_keypoints);
        imshow("SURF_Keypoints", img_surf_keypoints);

        // Detect the keypoints using SIFT Detector
        cv::SiftFeatureDetector sift_detector;
        std::vector<cv::KeyPoint> sift_keypoints;
        sift_detector.detect(frame, sift_keypoints);

        //Draw SIFT keypoints
        Mat img_sift_keypoints;
        cv::drawKeypoints(frame, sift_keypoints, img_sift_keypoints);

        //Show SIFT keypoints
        imshow("SIFT Keypoints", img_sift_keypoints);
    }

    // Releasing the resources and destroying the windows
    cap.release();
    destroyAllWindows();
    return 0;
}