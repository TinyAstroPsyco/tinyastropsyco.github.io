/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:11,February, 2024 
   DESCRIPTION:This C++ code leverages OpenCV to conduct content-based image retrieval. It calculates color and texture histograms for a target image and compares them with a dataset of images 
   using Euclidean distance. The top three matching images from the dataset, based on combined color and texture similarities, are then identified and displayed.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip> // for std::setw and std::setfill

/*
 Input: Takes a constant reference to a cv::Mat object, which represents an image.
 Description: Calculates a 2D histogram for color based on the input image in the HSV color space.
 Output: Returns the calculated 2D histogram as a cv::Mat object.
*/
cv::Mat calculateHistogram2D(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    
    return hist;
}

/*
 Input: Takes a constant reference to a cv::Mat object, which represents an image.
 Description: Calculates a texture histogram using the Sobel operator to find gradients in the input grayscale image.
 Output: Returns the calculated texture histogram as a cv::Mat object.
*/
cv::Mat calculateTextureHistogram(const cv::Mat& image) {
    cv::Mat gray, sobelX, sobelY, gradientMagnitude;

    // Convert to grayscale
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Apply Sobel operator to find gradients in x and y directions
    cv::Sobel(gray, sobelX, CV_32F, 1, 0);
    cv::Sobel(gray, sobelY, CV_32F, 0, 1);
    cv::magnitude(sobelX, sobelY, gradientMagnitude);

    // Calculate histogram of gradient magnitudes
    int histSize[] = {64}; // Adjust size as needed
    float range[] = {0, 256};
    const float* histRange = {range};

    cv::Mat hist;
    cv::calcHist(&gradientMagnitude, 1, 0, cv::Mat(), hist, 1, histSize, &histRange, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return hist;
}

/*
 Input: Takes two constant references to cv::Mat objects, which represent histograms.
 Description: Calculates the Euclidean distance between two histograms.
 Output: Returns the Euclidean distance as a float value.
*/
float calculateEuclideanDistance(const cv::Mat& hist1, const cv::Mat& hist2) {
    return cv::norm(hist1, hist2, cv::NORM_L2);
}

int main() {
    // Load the target image
    cv::Mat targetImage = cv::imread("C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\pic.0535.jpg");

    if (targetImage.empty()) {
        std::cerr << "Error: Unable to load target image" << std::endl;
        return -1;
    }

    // Calculate histograms for the target image
    cv::Mat targetColorHist = calculateHistogram2D(targetImage);
    cv::Mat targetTextureHist = calculateTextureHistogram(targetImage);

    std::vector<std::pair<float, std::string>> distances;

    // Directory containing dataset images
    std::string datasetDir = "C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\";

    // Loop over all images in the dataset
    for (int i = 1; i <= 1107; ++i) { // Adjust based on your dataset size
        std::stringstream filenameStream;
        filenameStream << datasetDir << "pic." << std::setw(4) << std::setfill('0') << i << ".jpg";
        std::string filename = filenameStream.str();

        cv::Mat currentImage = cv::imread(filename);
        if (currentImage.empty()) continue;

        // Calculate histograms for the current image
        cv::Mat currentColorHist = calculateHistogram2D(currentImage);
        cv::Mat currentTextureHist = calculateTextureHistogram(currentImage);

        // Compute distances
        float colorDistance = calculateEuclideanDistance(targetColorHist, currentColorHist);
        float textureDistance = calculateEuclideanDistance(targetTextureHist, currentTextureHist);
        float combinedDistance = (colorDistance + textureDistance) / 2.0f; // Average distance

        distances.emplace_back(combinedDistance, filename);
    }

    // Sort based on combined distance
    std::sort(distances.begin(), distances.end());

    std::cout << "Top 3 matching images:" << std::endl;
    for (int i = 1; i < std::min(3, static_cast<int>(distances.size()))+1; ++i) {
        std::cout << "Match " << i + 1 << ": Distance: " << distances[i].first << ", Filename: " << distances[i].second << std::endl;
    }

    return 0;
}