/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:18,March, 2024
   DESCRIPTION: The code reads the intrinsic camera parameters from a yml file and then proceeds to calculate the current position of the camera, project the corners on to the image plane
   and draws virtual objects on the frame.
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

		/*
		//Displaying a green dot for verifying if the projection_image_points are being projected at the right place
		for (Point i : projection_image_points) {
			cout << "Point x : " << i.x << "Point y : " << i.y << endl;
			circle(frame, Point(i.x, i.y), 2, Scalar(0, 0, 255), -1, LINE_AA);
			//putText(frame, "Projected point check with object points ", Point(40, 20), 3, 0.5, Scalar(0, 0, 255), 1, 8, false);
			//putText(frame, string(to_string(point_id)), Point(i.x, i.y), 3, 0.5, Scalar(255, 0, 0), 1, 8, false);

			//Displaying the axis alternatively
			//arrowedLine(frame, Point(i.x, i.y), Point(i.x + 20, i.y + 0), Scalar(0, 255, 0), 1, 8, 0.2);
			//arrowedLine(frame, Point(i.x, i.y), Point(i.x + 0, i.y - 20), Scalar(0, 0, 255), 1, 8, 0.2);
			point_id++;
		}
		*/

		
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




//Main code starts here
int main() {
	Mat frame;//Initilizing the image frame
	Size patternSize(9, 6);//Defining the number of internal corners
	vector<Point2f> corner_set; //Image Points
	vector<Vec3f> point_set; //Object Points
	vector<vector<Vec3f>> point_list; //To store Object Points
	vector<vector<Point2f>> corner_list; //Image points in a vector
	VideoCapture cap(0);//Capturing the frame for laptops webcam

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


	for (;;) {

		cap.read(frame);
		if (waitKey(1) == 27) {
			break;
		}
		if (frame.empty()) {
			cout << "The frame is empty, please check your video feed" << endl;
			continue;
		}
		
		//Computing the position of the camera
		calculate_current_position_of_the_camera(frame, point_set, corner_set, K, k, patternSize, rvecs, tvecs);
		//Printing the sizes for debugging
		cout << "Size of the rvecs : " << rvecs.size() << "Size of tvecs : " << tvecs.size() << endl;
		//Projecting the points
		project_outside_corners(frame, rectangle_points, rvecs, tvecs, K, k);
		//Displaying the image
		imshow("Corners", frame);

	}
	cap.release();
	return 0;
}
