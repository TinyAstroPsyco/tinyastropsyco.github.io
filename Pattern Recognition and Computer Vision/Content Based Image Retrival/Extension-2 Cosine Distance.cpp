
/*
    STUDENT NAME: Poojit Maddineni (NUID: 002856550) ; Venkata Satya Naga Sai Karthik Koduru (NUID: 002842207)
    DATE: 02-11-2024
    DESC: This program takes takes a target image and the feature vectors from a external csv file and then returns the top 3 mathces from the dataset.
*/

//Importing the librarires
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric> 
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;


// Function to check if a string ends with a given suffix for example ".jpj" and ".png"
bool endsWith(const string& str, const string& suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

/*
Input: Takes 1 argument, the data set directory path.
Description: Checks for the image name ending with .jpg, .png and etc and appends to a varibale to hold every image path
Output: gives a vector containing the image path strings.
*/


// Function to fetch image paths from a directory
vector<string> fetchImagePaths(const string& directoryPath) {
    vector<string> imagePaths;

    // Prepare search pattern
    string searchPattern = directoryPath + "\\*.*";

    WIN32_FIND_DATAA fileData;
    HANDLE hFind = FindFirstFileA(searchPattern.c_str(), &fileData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                string filePath = directoryPath + "\\" + fileData.cFileName;

                // Check if the file has an image extension
                if (endsWith(filePath, ".jpg") || endsWith(filePath, ".png") || endsWith(filePath, ".ppm") || endsWith(filePath, ".tif")) {
                    imagePaths.push_back(filePath);
                }
            }
        } while (FindNextFileA(hFind, &fileData) != 0);

        FindClose(hFind);
    }
    else {
        cout << "Cannot open directory: " << directoryPath << endl;
    }

    return imagePaths;
}




/*
Input: Takes 1 argument, the 7x7 square portion of the image.
Description: computes feature vector.
Output: Returns feature vector.
*/
vector<float> extractFeatureVector(const Mat& squareBox) {
    vector<float> featureVector;

    // Ensure the square box has 3 channels and is 7x7
    if (squareBox.channels() == 3 && squareBox.rows == 7 && squareBox.cols == 7) {
        // Flatten the 3-channel matrix into a 1D vector
        for (int c = 0; c < squareBox.channels(); ++c) {
            for (int i = 0; i < squareBox.rows; ++i) {
                for (int j = 0; j < squareBox.cols; ++j) {
                    featureVector.push_back(static_cast<float>(squareBox.at<Vec3b>(i, j)[c]));
                }
            }
        }
    }

    return featureVector;
}



/*
Input: Takes 1 argument, all the image paths.
Description: Computes the feature vector.
Output: returns the feature vector
*/
vector<vector<float>> computeFeatureVectors(const vector<string>& imagePaths) {
    vector<vector<float>> allFeatureVectors;

    // Loop through each image path
    for (const auto& imagePath : imagePaths) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Unable to read image " << imagePath << endl;
            continue;
        }

        // Extract the 7x7 square box from the center of the image
        int centerX = image.cols / 2;
        int centerY = image.rows / 2;
        Rect roi(centerX - 3, centerY - 3, 7, 7); // 7x7 square box
        Mat squareBox = image(roi);

        // Extract 3-channel feature vector from the 7x7 square box
        vector<float> featureVector = extractFeatureVector(squareBox);

        // Add the feature vector to the list
        allFeatureVectors.push_back(featureVector);
    }

    return allFeatureVectors;
}



/*
Input: Takes 3 arguments, the file name of csv, image paths and the feature vector for the entire dataset
Description: weites a csv file in the location.
Output: none.
*/
void writeCSV(const std::string& filename,
    const std::vector<std::string>& imagePaths,
    const std::vector<std::vector<float>>& featureVectors) {
    std::ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing" << std::endl;
        return;
    }

    // Write the header line
    outputFile << "Image Path,Feature Vector" << std::endl;

    // Iterate through image paths and feature vectors
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        // Write the image path
        outputFile << imagePaths[i] << ",";

        // Write the feature vector
        for (size_t j = 0; j < featureVectors[i].size(); ++j) {
            outputFile << featureVectors[i][j];
            if (j < featureVectors[i].size() - 1) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;
    }

    outputFile.close();
}


// Structure to hold image data
struct ImageData {
    std::string imagePath;
    std::vector<float> featureVector;
};



/*
Input: Takes 1 argument, the path of the csv file.
Description: Reads the csv file and appends the file name and the corresponding feature vector to the return variable.
Output: returns concatinated vector containing list of filenames and featurevectors.
*/
// Function to read data from CSV file
std::vector<ImageData> readCSV(const std::string& filename) {
    std::vector<ImageData> imageData;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for reading" << std::endl;
        return imageData;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string imagePath;
        std::vector<float> featureVector;

        // Read image path
        std::getline(iss, imagePath, ',');

        // Read feature vector
        float value;
        while (iss >> value) {
            featureVector.push_back(value);
            if (iss.peek() == ',') {
                iss.ignore();
            }
        }

        // Create ImageData instance and add to vector
        imageData.push_back({ imagePath, featureVector });
    }

    file.close();
    return imageData;
}


/*
Input: Takes 2 arguments, the feature vectors of target image and the other comparision image.
Description: calculates the square sum of differences.
Output: gives out the distance.
*/
/*
// Function to calculate distance using sum-square distance metric between two feature vectors
float calculateDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {

    // Check for empty vectors
    if (vec1.empty() || vec2.empty()) {
        std::cerr << "Error: Empty vectors provided" << std::endl;
        return -1.0; // Return an error value
    }

    float distance = 0.0;

    for (size_t i = 0; i < vec1.size(); ++i) {
        distance += std::pow(vec1[i] - vec2[i], 2);
        //cout << "Vec1=" << vec1[i];
    }
    return distance;
}
*/

// Function to calculate the cosine distance between two vectors
double calculateDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    double dotProduct = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
    double v1Norm = std::sqrt(std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0));
    double v2Norm = std::sqrt(std::inner_product(vec2.begin(), vec2.end(), vec2.begin(), 0.0));
    return (1.0 - (dotProduct / (v1Norm * v2Norm)));
}


/*
Input: Takes 2 arguments, the target image path and the image dataset containing paths and feature vectors.
Description: finds the taret images feature vector from the list.
Output: Returns the feature vector of the target image.
*/
// Function to find and return the feature vector of the target image
std::vector<float> findTargetFeatureVector(const std::string& targetImagePath, const std::vector<ImageData>& imageData) {
    for (const auto& data : imageData) {
        if (data.imagePath == targetImagePath) {
            return data.featureVector;
        }
    }
    // If target image path is not found, return an empty vector
    return std::vector<float>();
}



/*
Input: Takes 1 argument, the path of the csv file.
Description: Reads the csv file and appends the file name and the corresponding feature vector to the return variable.
Output: returns concatinated vector containing list of filenames and featurevectors.
*/

// Function to read feature vectors from the CSV file
std::unordered_map<std::string, std::vector<float>> readFeatureVectorsFromCSV(const std::string& filename) {
    std::unordered_map<std::string, std::vector<float>> featureVectors;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for reading" << std::endl;
        return featureVectors;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string filename;
        float value;
        std::vector<float> vector;
        iss >> filename;
        while (iss >> value) {
            vector.push_back(value);
        }
        featureVectors[filename] = vector;
    }
    file.close();
    return featureVectors;
}

// Comparator function to sort pairs based on the second element (the distance)
bool comparePairs(const std::pair<std::string, float>& pair1, const std::pair<std::string, float>& pair2) {
    return pair1.second < pair2.second; // Compare the second element of the pairs
}






//Main Code Starts Here
//Displays the top 3 matches of the target image.

int main() {


    //Question 5 main code starts here::

    string q5Path = "C:\\Users\\pooji\\Downloads\\ResNet18_olym.csv";
    std::vector<ImageData> q5Data = readCSV(q5Path);
    size_t count = 0; // Counter variable to track the number of iterations
    vector<float> distanceMetric;

    for (const auto& path : q5Data) {
        if (count >= 3) { // Check if we have already printed 3 elements
            break; // Exit the loop
        }
        cout << "Data for q 5 : " << path.imagePath << " ";
        // Print the first 2 elements of the featureVector
        for (float vals : path.featureVector) {
            cout << path.featureVector[vals] << " ";
            distanceMetric.push_back(path.featureVector[vals]);
        }
        cout << endl;
        cout << endl;


        ++count; // Increment the counter
    }




    // Find and return the feature vector of the target image for Q5
    string q5targetImagePath = "pic.0893.jpg";

    std::vector<float> q5targetFeatureVector = findTargetFeatureVector(q5targetImagePath, q5Data);
    // Print the target feature vector
    if (!q5targetFeatureVector.empty()) {
        std::cout << "Feature vector of the target image:" << std::endl;
        for (float q5value : q5targetFeatureVector) {
            std::cout << q5value << " ";
        }
        std::cout << std::endl;
    }
    else {
        std::cerr << "Target image path not found in the CSV file." << std::endl;
    }

    // Calculate distance metric for each image and print the result

    vector<pair<string, float>> q5eucledian_distances;

    for (const auto& q5data : q5Data) {
        float q5distance = calculateDistance(q5targetFeatureVector, q5data.featureVector);
        std::cout << "Distance between target image and " << q5data.imagePath << ": " << q5distance << std::endl;
        q5eucledian_distances.push_back(make_pair(q5data.imagePath, q5distance));

    }





    // Sort the vector in ascending order based on the distance
    std::sort(q5eucledian_distances.begin(), q5eucledian_distances.end(), comparePairs);

    // Print the sorted distances
    for (const auto& distance : q5eucledian_distances) {
        std::cout << "Distance: " << distance.second << " for image " << distance.first << std::endl;
    }

    cout << "The Top3 Matches for the target image are:" << endl;

    // Print the file names of the first four elements
    int q5count = 0;
    for (const auto& q5distance : q5eucledian_distances) {
        std::cout << "Image Path: " << q5distance.first << std::endl;
        ++q5count;
        if (q5count == 4) {
            break; // Exit the loop after printing the first four elements
        }
    }


    //Displaying the first 4 mathches
    for (int q5disp = 0; q5disp < 4 && q5disp < q5eucledian_distances.size(); ++q5disp) {
        // Load the image from the file path
        cv::Mat image = cv::imread("C:\\Users\\pooji\\Downloads\\Test_data\\" + q5eucledian_distances[q5disp].first);

        // Check if the image is loaded successfully
        if (!image.empty()) {
            // Display the image
            cv::imshow("Extension - 2", image);
            cv::waitKey(0); // Wait for a key press
        }
        else {
            std::cerr << "Error: Unable to load image " << q5eucledian_distances[q5disp].first << std::endl;
        }
    }


    return 0;
}



