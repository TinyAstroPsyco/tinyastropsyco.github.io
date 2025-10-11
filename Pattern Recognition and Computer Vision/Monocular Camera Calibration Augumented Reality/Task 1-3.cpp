/*
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:18,March, 2024
   DESCRIPTION: This code is used to calibrate the camera and stores the intrinsic parameters in a file.
*/


//Importing then necessary headers
#include <iostream>
#include <stdio.h>
#include <vector>
#include<opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

//Name space declarations
using namespace std;
using namespace cv;


/*
 Description: Function to generate 3D world coordinates for a given pattern size.
 Input: Takes in the pattern size of the checker board, and the empty point set to store the object points
 Output: Populates the object points in point_set based on the pattern size
*/
void generate_3d_points(vector<Vec3f>& point_set, Size& patternSize) {
	//Iterating over the rows and colums to assign the world coordinate points
	for (int x = 0; x < patternSize.height; ++x) {
		for (int y = 0; y < patternSize.width; ++y) {
			//Defining worldPoint to store the object point
			Vec3f worldPoint;
			//Considering the sqauare side distance is 1 unit
			if (x != 0) {
				worldPoint[0] = y * 1;
				worldPoint[1] = x * (-1);
				worldPoint[2] = 0;
			}
			else {
				worldPoint[0] = y * 1;
				worldPoint[1] = x * 1;
				worldPoint[2] = 0;
			}
			//Appending the worldpoints to the point set to store it
			point_set.push_back(worldPoint);
		}
	}
}



/*
 Description: Function to detect and extract corners from an input image frame.
 Task: Task - 1
 Input: Takes the image frame, a vector to store corner points, and the pattern size.
 Output: The corener points on the image are appended to the corner_set and also the points are visualized by drawing a circle at the location the x and y axis and the point id

*/void detect_and_extract_target_corners(Mat& frame, vector<Point2f>& corner_set, Size& patternSize) {
	vector<Point2f> detected_corners;
	Mat gray;
	//Converting the image to grayscale to pass it to findChessboardCorners function
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	//Setting the flag values
	int flags = CALIB_CB_ADAPTIVE_THRESH;
	//finding if the pattern is detected
	bool pattern_found = findChessboardCorners(gray, patternSize, corner_set, flags);
	
	//Drawing the corners and appending the corner points if detected
	if (pattern_found) {
		detected_corners = corner_set;
		putText(frame, string("The number of corners detected = " + to_string(corner_set.size())), Point(25, 25), 3, 0.5, Scalar(255, 0, 0), 1, 8, false);
		putText(frame, string("The coordinate of the first point is = " + to_string(corner_set[0].x) + "," + to_string(corner_set[0].y)), Point(25, 50), 3, 0.5, Scalar(255, 0, 0), 1, 8, false);
		
		//Drawing the chess board corners
		int point_id = 0;//Initilizing the point id to mark the points on the image
		for (Point i : detected_corners) {
			circle(frame, Point(i.x, i.y), 4, Scalar(0, 0, 256), 2, 2, 0);
			putText(frame, string(to_string(point_id)), Point(i.x, i.y), 3, 0.5, Scalar(255, 0, 0), 1, 8, false);
			arrowedLine(frame, Point(i.x, i.y), Point(i.x + 20, i.y + 0), Scalar(0, 255, 0), 1, 8, 0.2);
			arrowedLine(frame, Point(i.x, i.y), Point(i.x + 0, i.y - 20), Scalar(0, 0, 255), 1, 8, 0.2);
			point_id++;
		}
	}
	//If not detected, specifies that the corners are not detected
	else {
		putText(frame, string("Corners not detected!!"), Point(25, 50), 3, 0.5, Scalar(255, 0, 0), 1, 8, false);
	}
}


/*
 Description: Function to calibrate the camera using detected corner points and world coordinates(object points).
 Task: task 3
 Input: Takes an input image frame, a vector of 3D world coordinates for corner points, and a vector of 2D corner points.
 Output: Outputs the calibrated camera matrix, distortion parameters, and stores them in a .yml file.
*/
void calibrate_camera(Mat& frame, vector<vector<Vec3f>>& point_list, vector<vector<Point2f>>& corner_list) {
	
	cout << "Camera Calibration Triggered" << endl;

	//Initilizing Camera Matrix
	Mat K = Mat::eye(3, 3, CV_64FC1);
	vector<double> k;//Initilizing the vector to store the distortion parameters
	K.at<double>(0, 2) = (frame.cols / 2);
	K.at<double>(1, 2) = (frame.rows / 2);

	//Printing the camera matrix before calibration
	cout << "Camera Matrix K Before Calibration:" << endl;
	for (int i = 0; i < K.rows; ++i) {
		for (int j = 0; j < K.cols; ++j) {
			cout << K.at<double>(i, j) << " ";
		}
		cout << endl;
	}
	int flags = cv::CALIB_FIX_ASPECT_RATIO;

	vector<Mat> rvecs, tvecs;//Initilizing the rvec and tvec

	//Calibrating the camera and finding the reprojection error
	float error = calibrateCamera(point_list, corner_list, frame.size(), K, k, rvecs, tvecs, flags);
	cout << "Reprojection Error = " << error << " Pixels" << endl;
	//Printing the camera matrix
	cout << "Camera Matrix K After Calibration:" << endl;
	for (int i = 0; i < K.rows; ++i) {
		for (int j = 0; j < K.cols; ++j) {
			cout << K.at<double>(i, j) << " ";
		}
		cout << endl;
	}
	cout << "Camera Distortion Parameters" << endl;
	for (int i : k) {
		cout << "The distortion value = " << i << endl;
	}
	//Storing the file parameters
	String calib_parameters = "D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\calibration_parameters.yml";
	FileStorage fs(calib_parameters, FileStorage::WRITE);
	fs << "Camera Parameters" << K;
	fs << "Camera Distortion Parameters" << k;
	fs.release();
}



//Main code starts here
int main() {
	Mat frame;//Initilizing the image frame
	Size patternSize(9, 6);//Defining the number of internal corners
	vector<Point2f> corner_set; //Image Points
	vector<Vec3f> point_set; //Object Points
	vector<vector<Vec3f>> point_list; //To store Object Points	
	int image_number = 0;
	vector<vector<Point2f>> corner_list; //Image points in a vector
	VideoCapture cap(0);
	//Generating the object points
	generate_3d_points(point_set, patternSize);//Generates the world coordinate points



	for (;;) {
		cap.read(frame);
		if (waitKey(1) == 27) {
			break;
		}
		if (frame.empty()) {
			cout << "The frame is empty, please check your video feed" << endl;
			continue;
		}

		detect_and_extract_target_corners(frame, corner_set, patternSize);//Detecting the corners
		imshow("Corners", frame);

		//Saving the frame when s is pressed
		if (waitKey(1) == 's') {

		//Storing the vector of corner set of the corresponding frame
		corner_list.push_back(corner_set);
		//Writing the frame to the location with the image number attached to image name
		imwrite("D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\image_" + to_string(image_number) + ".png", frame);//Saving the correspondig image
		image_number++;
		//Confirmation if the image is saved
		cout << "Saved the image number : " << image_number << endl;			
		//Appending the object points to the point list
		point_list.push_back(point_set);
		}

		//Calibrating the camera using the object and corresponding corner points
		if (waitKey(1) == 'c') {
			//Specifying a condition for the minimum number of images required for calibration
			if (image_number >= 5) {

				//Printing the image object points and the corresponding image corner points
				cout << "The stored Coordinates are -->" << endl;
				for (vector<Point2f> c : corner_list) {
					int pt_id = 0;
					for (Point2f p : c) {
						cout << "The x coordinate = " << p.x << " The y coordinate = " << p.y << "Point id = " << pt_id << endl;
						pt_id++;
					}
				}
				cout << "The stored World Coordinates are -->" << endl;
				for (vector<Vec3f> w : point_list) {
					int pt_id = 0;
					for (Vec3f w_p : w) {
						cout << "The x coordinate = " << w_p[0] << " The y coordinate = " << w_p[1] << " The z coordinate = " << w_p[2] << "Point id = " << pt_id << endl;
						pt_id++;
					}
				}
				//Calibrating the camera
				calibrate_camera(frame, point_list, corner_list);
				break;
			}
			else {
				cout << "The minimum number of images required for calibations are 5, this requirement is not met!!" << endl;;
				continue;
			}

		}
	}
	cap.release();
	return 0;
}
