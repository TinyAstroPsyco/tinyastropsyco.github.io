/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:18,March, 2024
   DESCRIPTION: This extension will make the pattern invisible by applying a custom mask on the pattern.
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



//Reading the calibration file containg the camera matix and the distortion paramenters
void read_calib_file(Mat& K, vector<double>& k) {

	string calib_file_name = "D:\\MS Robotics Neu\\Spring 2024\\PRCV\\Project-4-AR\\Resources\\calibration_parameters.yml";
	FileStorage fs(calib_file_name, FileStorage::READ);

	fs["Camera Parameters"] >> K; //Reading the Camera Parameters and storing the value into K
	fs["Camera Distortion Parameters"] >> k; //Reading the distortion parameters and appediing to k

	cout << "Camera Matrix K Readout:" << endl;
	for (int i = 0; i < K.rows; ++i) {
		for (int j = 0; j < K.cols; ++j) {
			cout << K.at<double>(i, j) << " ";
		}
		cout << endl;
	}
	cout << "Distortion Parameters Readout:" << endl;
	for (auto i : k) {
		//cout << "Distortion parameters : " << endl;
		cout << i << endl;

	}

	fs.release();
}



/*
 Description: Function to calculate the current position of the camera using provided 3D points and their corresponding 2D image points.
 Task: task 4
 Input: Takes an image frame, a list of 3D world points, a list of corresponding 2D corner points, camera matrix, distortion parameters, pattern size, rotation vectors, and translation vectors.
 Output: Modifies the rotation and translation vectors to represent the camera's current position.
*/
void calculate_current_position_of_the_camera(Mat& frame, vector<Vec3f>& point_set, vector<Point2f>& corner_set, Mat& K, vector<double> k, Size& patternSize, Mat& rvecs, Mat& tvecs) {
	bool useExtrinsicGuess = false;
	int flags = SOLVEPNP_ITERATIVE;
	//Ckecking if the pattern is found
	bool pattern_found = findChessboardCorners(frame, patternSize, corner_set, flags);
	/*
	for (int i = 0; i < K.rows; ++i) {
		for (int j = 0; j < K.cols; ++j) {
			cout << K.at<double>(i, j) << " ";
		}
		cout << endl;
	}
	*/

	//Finding the rvec and tvec
	if (pattern_found) {
		cout << "Pattern Found" << endl;
		solvePnP(point_set, corner_set, K, k, rvecs, tvecs, useExtrinsicGuess, flags);
		//cout << "Size of the rvecs : " << rvecs.size() << "Size of tvecs : " << tvecs.size() << endl;
	}
}


/*
 Description: Function to project 3D world points to the image plane.
 Task: task 5
 Input: Takes an image frame, a list of 3D points, rotation and translation vectors, camera matrix, distortion parameters, and a vector to store projected 2D points.
 Output: Modifies the input vector to contain the projected 2D points and draws lines between them on the input frame.
*/
void project_outside_corners(Mat& frame, vector<Vec3f>& point_set, Mat& rvecs, Mat& tvecs, Mat& K, vector<double> k) {


	vector<Point2f> projection_image_points;


	//Calculating the projections
	if (!rvecs.empty() && !tvecs.empty()) {
		projectPoints(point_set, rvecs, tvecs, K, k, projection_image_points);
		//cout << "The size of projection_image_points : " << projection_image_points.size() << endl;
		int point_id = 0;		
		for (size_t i = 0; i < projection_image_points.size(); ++i) {
			for (size_t j = i + 1; j < projection_image_points.size(); ++j) {
				Point p1(projection_image_points[i].x, projection_image_points[i].y);
				Point p2(projection_image_points[j].x, projection_image_points[j].y);
				line(frame, p1, p2, Scalar(0, 255, 0), 1, LINE_AA);
			}
		}
		//Clearing the rvecs and tvecs to eliminate the projection of points after the pattern is removed
		rvecs.release();
		tvecs.release();
	}
}



//Task 6 Creating different virtual objects

vector<Vec3f> square_points = {
	{0,0,0},
	{0,0,2},
	{2,0,0},
	{2,0,2},
	{2,2,0},
	{2,2,2},
	{0,2,2},
	{0,2,0}
};

vector<Vec3f> train_points = {
	// Train body
	{0, 0, 0},     // Front bottom left
	{3, 0, 0},     // Front bottom right
	{3, 2, 0},     // Front top right
	{0, 2, 0},     // Front top left
	{0, 0, -1},    // Back bottom left
	{3, 0, -1},    // Back bottom right
	{3, 2, -1},    // Back top right
	{0, 2, -1},    // Back top left
	// Train cabin
	{0.5, 1, 0},   // Cabin front bottom left
	{1.5, 1, 0},   // Cabin front bottom right
	{1.5, 2, 0},   // Cabin front top right
	{0.5, 2, 0},   // Cabin front top left
	{0.5, 1, -1},  // Cabin back bottom left
	{1.5, 1, -1},  // Cabin back bottom right
	{1.5, 2, -1},  // Cabin back top right
	{0.5, 2, -1}   // Cabin back top left
};


//Virtual object dodecahedron
std::vector<Vec3f> dodecahedron_points = {
	// Vertices of a regular dodecahedron
	// Each face is a regular pentagon

	// Top pentagon
	{0, 1, 1.618},     // A
	{-1.618, 0, 1},    // B
	{1, -1.618, 0},    // C
	{1, -1.618, 0},    // D
	{1.618, 0, 1},     // E

	// Bottom pentagon
	{0, -1, -1.618},   // F
	{1.618, 0, -1},    // G
	{1, 1.618, 0},     // H
	{-1, 1.618, 0},    // I
	{-1.618, 0, -1},   // J

	// Mid pentagons
	{0, -1.618, 1},    // K
	{1, 0, 1.618},     // L
	{1.618, 1, 0},     // M
	{0, 1.618, -1},    // N
	{-1.618, 1, 0}     // O
};



std::vector<Vec3f> rectangle_points = {
		{0, 0, 0},     // Vertex 0
		{10, 0, 0},    // Vertex 1
		{10, -6, 0},    // Vertex 2
		{0, -6, 0},     // Vertex 3
		{0, 0, 5},     // Vertex 4
		{10, 0, 5},    // Vertex 5
		{10, -6, 5},    // Vertex 6
		{0, -6, 5}      // Vertex 7
};




/*
 Description: Function to overlay a foreground image onto a background image at a specified location.
 Task: Extension 1
 Input: Takes the background image, foreground image, output image, and the location to place the foreground image.
 Output: Modifies the output image with the overlaid foreground image.
*/
void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2i location) {
	// Make sure the foreground image is smaller than the background and the location is valid
	if (location.x < 0 || location.y < 0 || location.x + foreground.cols > background.cols || location.y + foreground.rows > background.rows) {
		cerr << "Invalid location or foreground size!" << endl;
		return;
	}

	// Create a region of interest (ROI) for the overlay area
	Mat roi(output, Rect(location.x, location.y, foreground.cols, foreground.rows));

	// Overlay the foreground image onto the background using the alpha channel (if exists)
	if (foreground.channels() == 4) {
		// Separate the foreground alpha channel
		vector<Mat> channels;
		split(foreground, channels);

		// Copy the RGB channels of the foreground image to the ROI of the background
		Mat mask = channels[3]; // Alpha channel
		Mat mask_inv;
		bitwise_not(mask, mask_inv); // Invert the alpha channel

		// Create masked foreground and background images
		Mat fg, bg;
		bitwise_and(foreground, foreground, fg, mask); // Apply the mask to the foreground
		bitwise_and(roi, roi, bg, mask_inv); // Apply the inverted mask to the ROI of the background

		// Combine the masked foreground and background
		add(bg, fg, roi);
	}
	else {
		// If the foreground image does not have an alpha channel, directly overlay it onto the background
		foreground.copyTo(roi);
	}
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
	//Initilizzing the camera matrix
	Mat K;
	vector<double> k;
	//Initilizzing rvecs and tvecs
	Mat rvecs;
	Mat tvecs;
	//Generating the object points
	generate_3d_points(point_set, patternSize);//Generates the world coordinate points
	//Reading the intrinsic parameters 
	read_calib_file(K, k);
	bool apply_mask = false; // Flag to check if mask should be applied

	for (;;) {

		cap.read(frame);
		if (waitKey(1) == 27) {
			break;
		}
		if (frame.empty()) {
			cout << "The frame is empty, please check your video feed" << endl;
			continue;
		}
		// Check for keypress events
		char key = waitKey(1);
		
		//Computing the position of the camera
		calculate_current_position_of_the_camera(frame, point_set, corner_set, K, k, patternSize, rvecs, tvecs);
		//Printing the sizes for debugging
		cout << "Size of the rvecs : " << rvecs.size() << "Size of tvecs : " << tvecs.size() << endl;
		// Find the four extreme corners of the checkerboard
		vector<Point2f> extreme_corners;
		if (!corner_set.empty()) {
			extreme_corners.push_back(*min_element(corner_set.begin(), corner_set.end(), [](const Point2f& p1, const Point2f& p2) { return (p1.x + p1.y) < (p2.x + p2.y); }));
			extreme_corners.push_back(*max_element(corner_set.begin(), corner_set.end(), [](const Point2f& p1, const Point2f& p2) { return (p1.x - p1.y) < (p2.x - p2.y); }));
			extreme_corners.push_back(*min_element(corner_set.begin(), corner_set.end(), [](const Point2f& p1, const Point2f& p2) { return (p1.x - p1.y) < (p2.x - p2.y); }));
			extreme_corners.push_back(*max_element(corner_set.begin(), corner_set.end(), [](const Point2f& p1, const Point2f& p2) { return (p1.x + p1.y) < (p2.x + p2.y); }));
		}
		// Create a mask covering the entire frame
		Mat mask(frame.size(), CV_8UC1, Scalar(255));
		// Create a mask for the region of interest (ROI)
		Mat roi_mask = Mat::zeros(frame.size(), CV_8UC1);
		if (extreme_corners.size() == 4 && apply_mask) { // Only apply the mask if 'm' key is pressed
			
			// Define the expanded ROI using the extreme corners
			Rect roi_rect = boundingRect(extreme_corners);
			roi_rect -= Point(50, 50); //Moving the origin to cover the pattern
			roi_rect += Size(140, 140); //Moving the bottom right corner to cover the pattern
			roi_rect &= Rect(0, 0, frame.cols, frame.rows); // Ensure the ROI remains within the frame boundaries

			// Draw the expanded ROI on the mask
			rectangle(roi_mask, roi_rect, Scalar(255), FILLED);

			// Apply the texture to the expanded ROI
			if (!corner_set.empty()) {
				// Load the texture image
				Mat texture = imread("C:\\Users\\pooji\\Downloads\\Capitan_America_shield.jpeg");
				if (texture.empty()) {
					cerr << "Failed to load texture image!" << endl;
					return -1;
				}
				// Resize the texture to fit the size of the expanded ROI
				resize(texture, texture, roi_rect.size());
				// Apply the texture to the expanded ROI using the mask
				texture.copyTo(frame(roi_rect), roi_mask(roi_rect));
			}
		}
		// Apply the texture to the original frame
		project_outside_corners(frame, rectangle_points, rvecs, tvecs, K, k);		
		
		if (key == 'm') {
			// Toggle the apply_mask flag when 'm' key is pressed
			apply_mask = !apply_mask;
		}
		// Display the frame with virtual objects
		imshow("Virtual Objects with mask", frame);
	}
	cap.release();
	return 0;
}
