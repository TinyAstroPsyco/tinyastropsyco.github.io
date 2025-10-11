/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:11,February, 2024 
   DESCRIPTION:This code implements image retrieval using Fourier features computed via Fourier Transform with OpenCV. It reads a target image, computes Fourier features, 
   compares them with features of images in a directory, calculates similarity scores based on squared differences, and displays the top 5 similar images along with 
   their distances.
*/



#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <windows.h>

using namespace std;
using namespace cv;

/*

 Input: Takes 1 argument of type cv::Mat.
 Description: Computes Fourier features from a given image using Fourier Transform.
 Output: Returns a cv::Mat containing the computed Fourier features.

*/
Mat computeFourierFeatures(const Mat& inputImage) {
    // Compute Fourier Transform
    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY); // Convert image to grayscale
    Mat paddedImage; // Expand input image to optimal size
    int m = getOptimalDFTSize(grayImage.rows);
    int n = getOptimalDFTSize(grayImage.cols);
    copyMakeBorder(grayImage, paddedImage, 0, m - grayImage.rows, 0, n - grayImage.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(paddedImage), Mat::zeros(paddedImage.size(), CV_32F)};
    Mat complexImage;
    merge(planes, 2, complexImage); // Add to the expanded another plane with zeros
    dft(complexImage, complexImage); // Fourier transform
    split(complexImage, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

    // Compute magnitude spectrum
    Mat magnitudeSpectrum;
    magnitude(planes[0], planes[1], magnitudeSpectrum); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    magnitudeSpectrum += Scalar::all(1);
    log(magnitudeSpectrum, magnitudeSpectrum);

    // Resize the power spectrum to 16x16 image
    resize(magnitudeSpectrum, magnitudeSpectrum, Size(16, 16));

    // Normalize the magnitude spectrum
    normalize(magnitudeSpectrum, magnitudeSpectrum, 0, 1, NORM_MINMAX);

    return magnitudeSpectrum.reshape(0, 1);  // Flatten to a single row
}

/*

 Input: Takes 2 arguments of type cv::Mat.
 Description: Computes the sum of squared differences between two 16x16 feature matrices.
 Output: Returns the sum of squared differences as a double value.

*/
double calculateSumOfSquaredDifference(const Mat& featureMatrix1, const Mat& featureMatrix2) {
    double ssd = 0.0;
    for (int i = 0; i < featureMatrix1.rows; ++i) {
        for (int j = 0; j < featureMatrix1.cols; ++j) {
            ssd += pow(featureMatrix1.at<float>(i, j) - featureMatrix2.at<float>(i, j), 2);
        }
    }
    return ssd;
}

/*

 Input: Takes 1 argument of type std::string.
 Description: Lists files in a directory specified by the input path.
 Output: Returns a vector containing the names of files in the directory.

*/
vector<string> listFilesInDirectory(const string& datasetDir) {
    vector<string> files;
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA((datasetDir + "/*").c_str(), &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (findFileData.dwFileAttributes != FILE_ATTRIBUTE_DIRECTORY) {
                files.push_back(datasetDir + "/" + findFileData.cFileName);
            }
        } while (FindNextFileA(hFind, &findFileData) != 0);
        FindClose(hFind);
    }
    return files;
}

int main() {
    // Read target image
    Mat targetImage = imread("C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\pic.0984.jpg");
    if (targetImage.empty()) {
        cerr << "Error: Could not read target image." << endl;
        return 1;
    }

    // Compute features of target image
    Mat targetFeatures = computeFourierFeatures(targetImage);

    // Loop over the directory of images
    string datasetDir = "C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\";
    vector<pair<double, string>> similarityScores;

    // List files in the directory
    vector<string> files = listFilesInDirectory(datasetDir);
    for (const string& filePath : files) {
        // Read image
        Mat image = imread(filePath);
        if (image.empty()) {
            cerr << "Error: Could not read image " << filePath << endl;
            continue;
        }
        // Compute features of current image
        Mat imageFeatures = computeFourierFeatures(image);
        // Compute similarity score
        double distance = calculateSumOfSquaredDifference(targetFeatures, imageFeatures);
        // Store similarity score and image path
        similarityScores.push_back(make_pair(distance, filePath));
    }

    // Sort the list of matches
    sort(similarityScores.begin(), similarityScores.end());

    // Return top N matches (here, N = 5)
    int topN = 5;
    cout << "Top " << topN << " similar images:" << endl;
    for (int i = 0; i < min(topN, static_cast<int>(similarityScores.size())); ++i) {
        cout << similarityScores[i].second << " - Distance: " << similarityScores[i].first << endl;
        // Display similar images
        Mat similarImage = imread(similarityScores[i].second);
        if (!similarImage.empty()) {
            imshow("Similar Image " + to_string(i+1), similarImage);
            waitKey(0); // Wait for any key press
        }
    }

    return 0;
}
