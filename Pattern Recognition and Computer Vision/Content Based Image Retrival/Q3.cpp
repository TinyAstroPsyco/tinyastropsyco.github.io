/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:11,February, 2024 
   DESCRIPTION: This program splits the target image into top and bottom halves, calculates histograms for each half, compares them with dataset images, 
   combines intersection values, and displays the top 4 matching images.
*/



#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

/*
 Description: Function to calculate histogram intersection between two histograms.
 Input:
    - hist1: First histogram.
    - hist2: Second histogram.
 Output: Returns the histogram intersection value between hist1 and hist2.
*/
double histogramIntersection(const Mat &hist1, const Mat &hist2) {
    return compareHist(hist1, hist2, HISTCMP_INTERSECT);
}

/*
 Description: Function to calculate histogram of an image.
 Input:
    - image: Input image for which histogram is to be calculated.
 Output: Returns the histogram of the input image.
*/
Mat calculateHistogram(const Mat &image) {
    Mat hist;
    int channels[] = {0, 1, 2};
    int histSize[] = {8, 8, 8};
    float range[] = {0, 256};
    const float* ranges[] = {range, range, range};
    calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
    return hist.reshape(1, 1);
}


/*
 Description: Entry point of the program. Loads a target image, calculates color histograms for the target image,
              computes histograms for images in a dataset, calculates combined distances between target and dataset images,
              and displays the top 3 matching images based on combined distances.
 Output: Returns 0 upon successful execution.
*/
int main() {
    // Hardcoded target image path
    string targetImage = "C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\pic.0274.jpg";

    // Load target image
    Mat targetImage = imread(targetImage);
    if(targetImage.empty()) {
        cout << "Could not open or find the target image" << endl;
        return -1;
    }

    // Split target image into top and bottom halves
    Mat topHalf = targetImage(Rect(0, 0, targetImage.cols, targetImage.rows / 2));
    Mat bottomHalf = targetImage(Rect(0, targetImage.rows / 2, targetImage.cols, targetImage.rows / 2));

    // Calculate histograms for top and bottom halves
    Mat topHistogram = calculateHistogram(topHalf);
    Mat bottomHistogram = calculateHistogram(bottomHalf);

    // Directory containing dataset images
    string datasetDir = "C:\\Users\\kodur\\OneDrive\\Desktop\\CV projects\\Assignments\\Dataset\\olympus\\";
    vector<String> imagePaths;

    // Get list of image paths in dataset directory
    glob(datasetDir + "*.jpg", imagePaths, false);

    // Store image paths and corresponding histogram intersection values
    vector<pair<string, double>> imageMatches;

    // Loop through image paths
    for (const auto &imagePath : imagePaths) {
        // Load image
        Mat image = imread(imagePath);
        if (image.empty()) {
            cout << "Could not open or find the image " << imagePath << endl;
            continue;
        }

        // Calculate histograms for top and bottom halves of each image
        Mat imageTopHalf = image(Rect(0, 0, image.cols, image.rows / 2));
        Mat imageBottomHalf = image(Rect(0, image.rows / 2, image.cols, image.rows / 2));
        Mat imageTopHistogram = calculateHistogram(imageTopHalf);
        Mat imageBottomHistogram = calculateHistogram(imageBottomHalf);

        // Calculate histogram intersection with target histograms
        double intersectionTop = histogramIntersection(topHistogram, imageTopHistogram);
        double intersectionBottom = histogramIntersection(bottomHistogram, imageBottomHistogram);

        // Combine intersections using weighted averaging
        double combinedIntersection = 0.6 * intersectionTop + 0.4 * intersectionBottom;

        // Store image path and intersection value
        imageMatches.push_back({imagePath, combinedIntersection});
    }

    // Sort matches based on intersection values in descending order
    sort(imageMatches.begin(), imageMatches.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    // Output distances for the target image
    cout << "Distances for the target image " << targetImage << ":" << endl;
    for (const auto &match : imageMatches) {
        cout << match.first << " - Intersection: " << match.second << endl;
    }

    // Display the top 4 matches
    cout << "Top 4 matches for the target image:" << endl;
    for (int i = 0; i < min(4, static_cast<int>(imageMatches.size())); ++i) {
        Mat image = imread(imageMatches[i].first);
        imshow("Match " + to_string(i+1), image);
        waitKey(0);
    }

    return 0;
}

