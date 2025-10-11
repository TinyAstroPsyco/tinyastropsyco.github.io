
/*
    STUDENT NAME: Poojit Maddineni (NUID: 002856550) ; Venkata Satya Naga Sai Karthik Koduru (NUID: 002842207)
    DATE: 02-11-2024
    DESC: This program takes a target image and uses Histogram Matching to find the closest top 3 matches.
*/

//Importing the librarires


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>


/*
Input: Takes 1 argument, the image matrix.
Description: Calculates the histogram of the image matrix.
Output: Returns the histogram as a matrix.
*/

cv::Mat calculateHistogram2D(const cv::Mat& image) {
    // Convert image to HSV color space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Set histogram parameters
    float h_bins = 16.0f; //Bin size for h value
    int s_bins = 16; // Bin size for s value
    int histSize[] = { static_cast<int>(h_bins), s_bins };
    float h_ranges[] = { 0.0f, 180.0f };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };

    // Calculating the histogram
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::noArray(), hist, 2, histSize, ranges, true, false);

    // Normalizing the histogram
    hist /= (image.rows * image.cols);

    return hist;
}


/*
Input: Takes 2 arguments, the histogram matrices of target image and comparing image
Description: Calculates hte intersection distance.
Output: Returns the distance value.
*/

// Function to calculate Intersection distance between histograms
float calculateIntersectionDistance(const cv::Mat& hist1, const cv::Mat& hist2) {
    // Checking if histograms are of the same size
    if (hist1.rows != hist2.rows || hist1.cols != hist2.cols) {
        throw std::invalid_argument("Histogram dimension mismatch");
    }

    // Computing the intersection distance
    float intersection = 0.0f;
    for (int i = 0; i < hist1.rows; ++i) {
        for (int j = 0; j < hist1.cols; ++j) {
            float min_val = std::min(hist1.at<float>(i, j), hist2.at<float>(i, j));
            intersection += min_val;
        }
    }
    float total_bins = static_cast<float>(hist1.rows * hist1.cols);
    float intersectionDistance = 1.0f - (intersection / total_bins);
    return intersectionDistance;
}


//Main function, displays the top 3 matches.

int main() {
    // Load the target image
    cv::Mat targetImage = cv::imread("C:\\Users\\pooji\\Downloads\\Test_data\\pic.0164.jpg");

    // Check if the target image was loaded successfully
    if (targetImage.empty()) {
        std::cerr << "Error: Unable to load target image" << std::endl;
        return -1;
    }

    // Calculate 2D histogram of the target image
    cv::Mat targetHist = calculateHistogram2D(targetImage);

    // Define a data structure to store distances and filenames
    std::vector<std::pair<float, std::string>> distances;

    // Define the directory containing dataset images
    std::string datasetDir = "C:\\Users\\pooji\\Downloads\\Test_data\\";

    // Loop over all images in the dataset
    for (int i = 1; i <= 1107; ++i) { // Adjust the loop limits according to your dataset
        // Construct the filename with leading zeros
        std::stringstream filenameStream;
        filenameStream << datasetDir << "pic." << std::setw(4) << std::setfill('0') << i << ".jpg";
        std::string filename = filenameStream.str();

        // Load the current image
        cv::Mat currentImage = cv::imread(filename);

        // Check if the current image was loaded successfully
        if (currentImage.empty()) {
            std::cerr << "Error: Unable to load current image " << filename << std::endl;
            continue;
        }

        // Calculate 2D histogram of the current image
        cv::Mat currentHist = calculateHistogram2D(currentImage);

        // Compute Intersection distance between target and current histograms
        float distance = calculateIntersectionDistance(targetHist, currentHist);

        // Store the distance and filename in the data structure
        distances.push_back(std::make_pair(distance, filename));

        std::cout << "Checking Next file" << std::endl;
    }

    std::cout << "Sorted Distance" << std::endl;
    // Sort the images based on distances in ascending order
    std::sort(distances.begin(), distances.end());
    int q2count = 0;
    for (const auto& q2value : distances) {
        std::cout << "Q2 Image paths :" << q2value.second << " Distance:" << q2value.first << std::endl;
        if (q2count == 10) {
            break;
        }
        q2count++;
    }

    // Output the top three matches for the target image
    std::cout << "Top three matches for the target image pic.0164:" << std::endl;
    for (int i = 0; i < std::min(4, static_cast<int>(distances.size())); ++i) {
        std::cout << "Match" << i + 1 << ", Filename: " << distances[i].second << "Distance : " << distances[i].first << std::endl;
    }


    // Display the top three matches
    std::cout << "Displaying top three matches:" << std::endl;
    for (int i = 0; i < std::min(4, static_cast<int>(distances.size())); ++i) {
        cv::Mat imageToShow = cv::imread(distances[i].second);
        cv::imshow("Match " + std::to_string(i), imageToShow);
        cv::waitKey(0); // Wait for any key press
    }

    return 0;
}
