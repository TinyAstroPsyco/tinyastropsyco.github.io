/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:11,February, 2024 
   DESCRIPTION: This program calculates color and texture histograms for a target image, compares them with a dataset, and displays the top 3 matching images based on combined color and texture similarities. 
   It calculates color and texture histograms for a target image, compares them with a dataset, and displays the top 3 matching images based on combined color and texture similarities.
*/



#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip>

/*
 Input: Takes a constant reference to a cv::Mat, which represents the input image.
 Description: Converts the input image from BGR color space to HSV color space, calculates a 2D histogram for color in the HSV color space, and normalizes the histogram.
 Output: Returns the calculated 2D histogram as a cv::Mat.
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
 Input: Takes a constant reference to a cv::Mat, which represents the input image.
 Description: Applies a Gabor filter to the input image to extract texture features.
 Output: Returns the filtered image as a cv::Mat.
*/
cv::Mat applyGaborFilter(const cv::Mat& image, double theta, double lambda = 10.0) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    int kernel_size = 21;
    double sigma = 5.0, gamma = 0.5, psi = CV_PI * 0.5;
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma, psi, CV_32F);
    cv::Mat dest;
    cv::filter2D(grayImage, dest, CV_32F, kernel);
    return dest;
}

/*
 Input: Takes a constant reference to a cv::Mat, which represents the input image.
 Description: Calculates the histogram of a Gabor filter response.
 Output: Returns the calculated histogram as a cv::Mat.
*/
cv::Mat calculateGaborHistogram(const cv::Mat& image) {
    cv::Mat gaborResponse = applyGaborFilter(image, CV_PI / 4); // Applying Gabor filter

    // Convert Gabor response to a format suitable for histogram calculation
    gaborResponse.convertTo(gaborResponse, CV_8U, 255.0 / (std::numeric_limits<float>::max() - std::numeric_limits<float>::min()), 
                            -std::numeric_limits<float>::min() * 255.0 / (std::numeric_limits<float>::max() - std::numeric_limits<float>::min()));

    int histSize[] = {256};
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gaborResponse, 1, nullptr, cv::Mat(), hist, 1, histSize, &histRange, true, false);
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


/*
 Description: Entry point of the program. Loads a target image, calculates color and texture histograms for the target image, 
              computes histograms for images in a dataset, calculates combined distances between target and dataset images, 
              and displays the top 3 matching images based on combined distances.
 Output: Returns 0 upon successful execution.
*/
int main() {
    // Load the target image
    cv::Mat targetImage = cv::imread("C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\pic.0019.jpg");
    if (targetImage.empty()) {
        std::cerr << "Error: Unable to load target image." << std::endl;
        return -1;
    }

    // Calculate histograms for the target image
    cv::Mat targetColorHist = calculateHistogram2D(targetImage);
    cv::Mat targetTextureHist = calculateGaborHistogram(targetImage); // Gabor histogram

    std::vector<std::pair<float, std::string>> distances;
    std::string datasetDir = "C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\";

    // iterates over all the images in the data set, extra conditions added to look for images prefixed with 1,2&3 ZEROS
    for (int i = 1; i <= 1107; ++i) {
        std::stringstream filenameStream;
        if (i <= 9) {
            filenameStream << datasetDir << "pic.000" << i << ".jpg";
        } else if (i <= 99) {
            filenameStream << datasetDir << "pic.00" << i << ".jpg";
        } else if (i <= 999) {
            filenameStream << datasetDir << "pic.0" << i << ".jpg";
        } else {
            filenameStream << datasetDir << "pic." << i << ".jpg";
        }
        std::string filename = filenameStream.str();

        cv::Mat currentImage = cv::imread(filename);
        if (currentImage.empty()) continue;

        // Calculate histograms for the current image
        cv::Mat currentColorHist = calculateHistogram2D(currentImage);
        cv::Mat currentTextureHist = calculateGaborHistogram(currentImage); // Gabor histogram

        // Compute distances
        float colorDistance = calculateEuclideanDistance(targetColorHist, currentColorHist);
        float textureDistance = calculateEuclideanDistance(targetTextureHist, currentTextureHist);
        float combinedDistance = (colorDistance + textureDistance) / 2.0f; // Average distance

        distances.emplace_back(combinedDistance, filename);
    }

    // Sort based on combined distance (ascending order)
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Display top 3 matching images
    std::cout << "Top 3 matching images:" << std::endl;
    for (int i = 1; i < 4 && i < distances.size(); ++i) {
        std::cout << "Match1" << i + 1 << ": Filename: " << distances[i].second 
                  << ", Distance: " << distances[i].first << std::endl;
    }

    return 0;
}