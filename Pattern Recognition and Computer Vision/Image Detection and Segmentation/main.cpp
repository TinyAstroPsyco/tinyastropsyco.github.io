/*
   STUDENT NAME:poojit Maddineni (NUID:002856550)
   STUDENT NAME:Venkata Satya Naga Sai Karthik Koduru (NUID:002842207)
   DATE:11,February, 2024
   DESCRIPTION: This is a program to detect and classify objects based on their classical feature vectors.
   The program starts with a thresholded image, then applies dilation followed by erosion to enhance features. 
   This pre-processing step yields an eroded image. Subsequently, connected component analysis segments the image, isolating distinct objects. 
   Feature vectors are then extracted from these segmented regions, capturing essential characteristics. 
   These feature vectors are stored in a CSV file for easy access and analysis. Object detection relies on scaled Euclidean distance as a distance metric, 
   facilitating comparison between feature vectors for identification. Additionally, a confusion matrix is computed to evaluate the detection system's performance.
*/


//Importing the librarires
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>


//namespace usage declarations
using namespace cv;
using namespace std;



/*
 Description: Function to perform dilation on the input image.
 Input:Takes in a thresholdd binary image.
 Output: Returns the dilated image.
*/
void customDilate(const cv::Mat& src, cv::Mat& dst, int kernelSize) {
    dst = src.clone();//Cloning the src image
    //Iterating over the pixels in the source image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            uchar maxPixel = 0; //Initilizing the maximum pixel intensity to 0
            for (int ki = -kernelSize; ki <= kernelSize; ki++) {
                for (int kj = -kernelSize; kj <= kernelSize; kj++) {
                    // Calculate the new row and column indices for the neighborhood pixel
                    int ni = i + ki; // New row index
                    int nj = j + kj; // New column index
                    //checking if the new indices fall with in the bounds of the source image
                    if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                        maxPixel = std::max(maxPixel, src.at<uchar>(ni, nj));
                    }
                }
            }
            dst.at<uchar>(i, j) = maxPixel; //Assigning the maximum pixel intesnity value to the pixel in the destination image
        }
    }
}

/*
/*
 Description: Function to perform erosion on the input image.
 Input:Takes in the dilated image.
 Output: Returns the eroded image.
*/
void customErode(const cv::Mat& src, cv::Mat& dst, int kernelSize) {
    dst = src.clone();//Cloning the src image
    //Iterating over the pixels in the source image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            uchar minPixel = 255;//Initilizing the maximum pixel intensity to 255
            for (int ki = -kernelSize; ki <= kernelSize; ki++) {
                for (int kj = -kernelSize; kj <= kernelSize; kj++) {
                    // Calculate the new row and column indices for the neighborhood pixel
                    int ni = i + ki; // New row index
                    int nj = j + kj; // New column index
                    if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                        // Update minPixel with the minimum intensity found in the neighborhood
                        minPixel = std::min(minPixel, src.at<uchar>(ni, nj));
                    }
                }
            }
            // Only set the pixel values to 255 if neighbouring pixels have a value of 255
            dst.at<uchar>(i, j) = (minPixel == 255) ? 255 : 0;
        }
    }
}

//Task-1 and 2 is implemented here
/*
 Description: Function to convert the input image in to a binary image and then apply the dilation and erosion functions.
 Input: Takes the input image and the thresholding value as an argument.
 Output: Returns the eroded image.
*/
Mat preProcess_image(Mat& frame, int threshold_value, Mat& dilated_image, Mat& binary_image) {

    // Define the region of interest (ROI) using cv::Rect
    cv::Rect roi(0, 60, 600, 360); // (x, y, width, height)
    // Crop the frame using the ROI to eliminate unwanted edges on due to the camera setup
    frame = frame(roi);
    binary_image = frame.clone();
    cv::GaussianBlur(binary_image, binary_image, cv::Size(3, 3), 0, 0); //Applying a gaussian blur filter        
    //Converting the image into a binary image
    cvtColor(binary_image, binary_image, COLOR_BGR2GRAY);
    for (int i = 0; i < binary_image.rows; ++i) {
        for (int j = 0; j < binary_image.cols; ++j) {
            //Reading the pixel value at i,j
            uchar pixel = binary_image.at<uchar>(i, j);
            //Make the foreground white and the background black
            if (pixel <= threshold_value) {
                binary_image.at<uchar>(i, j) = 255;//if less than threshold
            }
            else {
                binary_image.at<uchar>(i, j) = 0;//else make them dark
            }
        }
    }

    Mat eroded_image;
    customDilate(binary_image, dilated_image, 1);//Dilating the image
    customErode(dilated_image, eroded_image, 1);//Eroding the image
    return eroded_image;

}




/*
 Description: Function to extract the individual region masks from the region map.
 Input: Takes the region map and the region id as the input.
 Output: Returns the region mask.
*/
Mat computeRegionFromRegionMap(const Mat& region_map, int region_id) {

    // Create a binary mask where pixels matching the region ID will be set to 255
    Mat regionMask;
    cv::compare(region_map, region_id, regionMask, cv::CMP_EQ);

    return regionMask;
}

/*
 Description: Structures to store the image data such as centroid values, regions ids and the area and also the image stastistics.
*/
struct ImageStats {
    Mat staistics;
};

struct ImageInfo {
    vector <int> regiod_id;
    vector<pair<double, double>> centroid;
    vector<double> area;
    vector<Mat> individual_region;
};


struct color_data {
    vector<int> region_id;
    Vec3b color;
    vector<pair<double, double>> centroid;
};


//Task-3 is implemented here

/*
 Description: This segments the image into individual regions and applies different colors to the different regions in the segmented image.
 Input: Takes the eroded image as the input from the preProcess_image function. The input image is named as binary_image, in the place of it, eroded image is passed.
 Output: Displays the segmented image and also returns the stastistics, the centroids and region ids of the regions in the segmented image.
*/

tuple<ImageInfo, Mat, Mat> segment_the_image_into_regions(Mat& binary_image, Mat& individual_segmented_image) {
    ImageInfo data; //Declaring data to hold the information about the centroids, region ids of the image
    color_data colornfo;
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary_image, labels, stats, centroids);
    unordered_map<int, int> region_mapping;
    int new_region_id = 0;//Initilizing the new region ids to 0
    //itreating thorugh each region to compute the area of the regions
    for (int n = 0; n < num_labels; ++n) {
        Mat labelMask = (labels == n);
        int region_area = countNonZero(labelMask);

        //Only considering areas above a threshold value, in this case it is 5000
        if (region_area > 5000) {
            data.regiod_id.push_back(new_region_id);//storing the region ids
            data.area.push_back(region_area);//storing the region areas
            double x_cord = centroids.at<double>(n, 0);
            double y_cord = centroids.at<double>(n, 1);
            data.centroid.push_back(make_pair(x_cord, y_cord));//Storing the corresponding centroids
            Mat individual_region = computeRegionFromRegionMap(labels, n);//seperating the individual region from the region map
            data.individual_region.push_back(individual_region);//Storing the corresponding region
            //reoderiing the region ids
            region_mapping[n] = new_region_id;
            ++new_region_id;
        }
    }

    // Iterate over each pixel in the labels matrix
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            // Get the original region ID at pixel (y, x)
            int original_region_id = labels.at<int>(y, x);
            // Look for the original region ID in the region mapping
            auto it = region_mapping.find(original_region_id);
            // Check if a mapping exists for the original region ID
            if (it != region_mapping.end()) {
                // If a mapping exists, update the region ID with the mapped ID
                labels.at<int>(y, x) = it->second;
            }
        }
    }

    // Create a color map assigment for visualizing different regions in the region map
    vector<Vec3b> colors(new_region_id);
    for (int i = 0; i < new_region_id; ++i) {
        colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256); // Generate random colors
    }

    // Create a color image to display the segmented regions
    Mat segmented_image(labels.size(), CV_8UC3, Scalar(0, 0, 0));

    // Assign colors to each pixel based on the region it belongs to
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            int original_region_id = labels.at<int>(y, x);
            auto it = region_mapping.find(original_region_id);
            if (it != region_mapping.end()) {
                segmented_image.at<Vec3b>(y, x) = colors[it->second];
            }
        }
    }

    individual_segmented_image = segmented_image;//Updating the individal region color
    return make_tuple(data, stats, labels);
}

//Task-4 is implemented here

/*
 Description: This computes the featurs for the regions in the segmented image.
 Input: Takes the original frame, segmented image, region number, centroid of the region and the region map of the segmented image.
 Output: Displays the centorid, the principal axis and the region number on the frame to visualize.
*/

void compute_features_for_region(Mat& frame, Mat binary_image, int& region_id, Point& center, Mat& region_map) {
    // Extract the region corresponding to the provided region ID
    Mat region = computeRegionFromRegionMap(region_map, region_id);

    // Find nding the non zero pixels in the segmeted image
    vector<Point> nonzero_points;
    findNonZero(region, nonzero_points);

    // Compute moments and features based on the region
    double m00 = nonzero_points.size();
    double m10_sum = 0.0, m01_sum = 0.0, mu20_sum = 0.0, mu02_sum = 0.0, mu11_sum = 0.0;
    for (const auto& coordinates : nonzero_points) {
        m10_sum += coordinates.x;
        m01_sum += coordinates.y;
    }

    // Calculate the center (centroid) of the region
    center.x = static_cast<int>(m10_sum / m00);
    center.y = static_cast<int>(m01_sum / m00);

    // Recompute moments based on the center
    mu11_sum = mu20_sum = mu02_sum = 0.0;
    for (const auto& coordinates : nonzero_points) {
        mu20_sum += (coordinates.x - center.x) * (coordinates.x - center.x);
        mu02_sum += (coordinates.y - center.y) * (coordinates.y - center.y);
        mu11_sum += (coordinates.x - center.x) * (coordinates.y - center.y);
    }

    // Calculate the orientation angle
    double theta = 0.5 * atan2(2 * mu11_sum, mu20_sum - mu02_sum);

    // Normalize and scale the principal axis for better representation
    cv::Point2f normalized_axis(cos(theta), sin(theta));
    cv::arrowedLine(frame, center, Point(center.x + 100 * normalized_axis.x, center.y + 100 * normalized_axis.y), Scalar(0, 255, 0), 2);

    // Convert region ID to string for display
    string region_id_string = to_string(region_id);

    // Display the region ID on the frame
    cv::putText(frame, region_id_string, Point(center.x, center.y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, LINE_8, false);
}






/*
Description: This function computes features for regions in the segmented image and visualizes them on the original frame and draws the oriented bounding box.
Input:
    - image_info: Information about the segmented regions including centroids and individual regions.
    - frame: The original frame where the regions and features will be visualized.
    - processed_image: The segmented image containing regions.
    - region_map: A map representing regions in the segmented image.
    - id: The ID of the region for which features are to be computed.
Output:
    - Returns a tuple containing the percentage filled, height-to-width ratio, centroid X-coordinate, and centroid Y-coordinate of the region.
    - The function also visualizes centroids, oriented bounding boxes, principal axes, and region numbers on the frame.
*/

tuple<double, double, double, double> find_feature_vectors_id(ImageInfo& image_info, Mat& frame, Mat& processed_image, Mat& region_map, int id) {
    double height_to_width_ratio = 0;
    double percentage_filled = 0;
    double centroid_x = 0;
    double centroid_y = 0;

    // Retrieve centroid coordinates for the given ID
    pair<double, double> centroid_coords = image_info.centroid[id];
    int x_int = static_cast<int>(centroid_coords.first);
    int y_int = static_cast<int>(centroid_coords.second);

    // Draw the centroid of the region
    cv::circle(frame, cv::Point(x_int, y_int), 5, cv::Scalar(0, 0, 255), -1);

    // Access the individual region corresponding to the given ID
    Mat individual_region = image_info.individual_region[id];

    // Find the contours of the individual region
    vector<vector<Point>> contours;
    findContours(individual_region, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        // Compute the minimum area rectangle around the contours
        RotatedRect rotated_rect = minAreaRect(contours[0]);

        // Convert double coordinates to integers
        Point centroid(x_int, y_int);

        // Compute features for the region
        compute_features_for_region(frame, processed_image, id, centroid, region_map);
        centroid_x = centroid.x;
        centroid_y = centroid.y;

        // Calculate height-to-width ratio and percentage filled
        double bbox_width = rotated_rect.size.width;
        double bbox_height = rotated_rect.size.height;

        // Ensuring the height always remains the same irrespective of rotation
        if (bbox_height < bbox_width) {
            swap(bbox_height, bbox_width); // Swap the values if bbox_width is larger
        }

        height_to_width_ratio = bbox_height / bbox_width;
        double contour_area = contourArea(contours[0]);
        double bounding_box_area = bbox_width * bbox_height;
        percentage_filled = (contour_area / bounding_box_area) * 100.0;

        // Convert the computed values to strings and draw on the image
        string ratio_text = "H/W Ratio: " + to_string(height_to_width_ratio);
        string percentage_text = "Filled: " + to_string(percentage_filled) + "%";
        cv::putText(frame, ratio_text, centroid, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, percentage_text, cv::Point(centroid.x, centroid.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }

    return make_tuple(percentage_filled, height_to_width_ratio, centroid_x, centroid_y);
}




/*
Description: Structure to store the information of the feature vectors
*/

// Struct to store the feature vectors for each region
struct RegionFeature {
    double percentage_filled;
    double height_to_width_ratio;
    string label;
};


// Struture to stor the label and the corresponding features
struct DataStruct {
    string label;
    double height_to_width_ratio;
    double percentage_filed;
};

/*
Description: Function to read the CSV file and store the data in a vector of DataStruct objects
Input: Takes the file path of the csv file
Output: Returns csv_data, a structure of stored features of the objects in the databases.
*/

std::vector<DataStruct> read_csv(const std::string filename) {
    std::vector<DataStruct> csv_data;//Inilitizing the structure
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return csv_data;
    }

    string line;
    getline(file, line); // skip header in the csv file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() >= 3) {
            DataStruct row;
            row.label = tokens[0];
            row.height_to_width_ratio = std::stod(tokens[1]); //Storing height to width ratio
            row.percentage_filed = std::stod(tokens[2]); //Storing percentage filled
            csv_data.push_back(row);//Storing the values in the file structure
            //cout << "Read the file " << endl;
        }
    }

    file.close();//Close the csv file
    return csv_data;
}


//Task-5 is partly implemented here and at other places

/*
Description: Function to save RegionFeature data to a CSV file.
Input: Takes a vector of RegionFeature objects as input, containing data such as percentage filled, height to width ratio, and label.
Output: Appends the RegionFeature data to a CSV file located at a specified file path.
*/
void save_to_csv(const vector<RegionFeature>& features) {
    // specify the file path here 
    string filePath = "D:\\MS Robotics Neu\\Spring 2024\\object_features.csv";

    // Open the CSV file in append mode
    ofstream outputFile(filePath, ios::app);
    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open the file for writing." << endl;
        return;
    }

    // Check if the file is empty
    outputFile.seekp(0, ios::end);
    if (outputFile.tellp() == 0) {
        // Write header to the CSV file
        outputFile << "Label,Height to Width Ratio,Percentage Filled" << endl;
    }

    // Write each RegionFeature to the CSV file
    for (const auto& feature : features) {
        outputFile << feature.label << "," << feature.height_to_width_ratio << "," << feature.percentage_filled << endl;
    }

    // Close the CSV file
    outputFile.close();

    cout << "The Label and the corresponding feature vectors are stored in : " << filePath << endl;
}


//Task-6 is partly implemented here and at other places

/*
Description: Computes the scaled Euclidean distance between a live feature vector and a stored feature vector.
Input: Takes a live feature vector represented by live_height_to_width_ratio and live_bounding_box_area,
       and a stored feature vector represented by feature_vector_data (DataStruct).
       Also takes standard deviations for height-to-width ratio and percentage filled (standastandardDeviation_heightToWidthRatio, standardDeviation_percentageFilled).
Output: Returns the scaled Euclidean distance between the live and stored feature vectors.
         Additionally, outputs information for debugging and comparison.
*/

double scaledEuclideanDistance(const DataStruct& feature_vector_data, double live_height_to_width_ratio, double live_bounding_box_area, double standastandardDeviation_heightToWidthRatio, double standardDeviation_percentageFilled) {
    double distance = 0.0;
    distance += ((live_bounding_box_area - feature_vector_data.percentage_filed) / standardDeviation_percentageFilled) * ((live_bounding_box_area - feature_vector_data.percentage_filed) / standardDeviation_percentageFilled);
    distance += ((live_height_to_width_ratio - feature_vector_data.height_to_width_ratio) / standastandardDeviation_heightToWidthRatio) * ((live_height_to_width_ratio - feature_vector_data.height_to_width_ratio) / standastandardDeviation_heightToWidthRatio);
    // No need to take square root as we only need to compare distances, so not computing the squareroot    
    //cout << "Live height to width ratio = " << live_height_to_width_ratio << " Live bounding box area = " << live_bounding_box_area << endl;
    //cout << "Height to width stored ratio = " << feature_vector_data.height_to_width_ratio << " Stored area = " << feature_vector_data.percentage_filed << endl;

    return distance;
}




/*
Description: Computes the standard deviations of features from a given vector of DataStruct objects.
Input: Takes a vector of DataStruct objects representing the feature vectors.
Output: Returns a tuple containing the standard deviations for height-to-width ratio and percentage filled.
         Additionally, provides insights into the data distribution and variability.
*/
tuple<double, double> computeFeatureStdDeviations(const std::vector<DataStruct>& feature_vector_data) {
    size_t numObjects = feature_vector_data.size();
    // Initialize sums for each feature
    double sumPercentFilled = 0.0, sumBoundingBoxRatio = 0.0;
    // Compute sums
    for (const auto& object : feature_vector_data) {
        sumPercentFilled += object.height_to_width_ratio;
        sumBoundingBoxRatio += object.percentage_filed;
    }

    // Compute means
    double meanPercentFilled = sumPercentFilled / numObjects;
    double meanBoundingBoxRatio = sumBoundingBoxRatio / numObjects;
    //std::cout << "Means: " << meanPercentFilled << ", " << meanBoundingBoxRatio << std::endl;

    // Initialize squared differences for each feature
    double squaredDiffPercentFilled = 0.0, squaredDiffBoundingBoxRatio = 0.0;

    // Compute squared differences
    for (const auto& object : feature_vector_data) {
        squaredDiffPercentFilled += (object.percentage_filed - meanPercentFilled) * (object.percentage_filed - meanPercentFilled);
        squaredDiffBoundingBoxRatio += (object.height_to_width_ratio - meanBoundingBoxRatio) * (object.height_to_width_ratio - meanBoundingBoxRatio);
    }
    //std::cout << "Squared Differences: " << squaredDiffPercentFilled << ", " << squaredDiffBoundingBoxRatio << std::endl;

    double standardDeviation_percentageFilled = std::sqrt(squaredDiffPercentFilled / numObjects);
    double standardDeviation_heightToWidthRatio = std::sqrt(squaredDiffBoundingBoxRatio / numObjects);
    return make_tuple(standardDeviation_percentageFilled, standardDeviation_heightToWidthRatio);
}



//*******************Task-9***************//


/*
Description: This function implements the k-nearest neighbors (KNN) matching algorithm to classify data points based on their nearest neighbors.
It calculates weights for each label based on distances, normalizes the weights, and returns the label with the highest weighted sum among the k-nearest neighbors.
Input:
- euclidean_distances: A vector of pairs representing the Euclidean distances between the data points and their corresponding labels.
- k: An integer representing the number of nearest neighbors to consider.
Output:
- Returns a string indicating the label that best matches the input data point based on the KNN algorithm.
*/

std::string knn_matching(const std::vector<std::pair<std::string, double>>& eucledian_distances, int k) {
    double weight_sum = 0.0;
    std::unordered_map<std::string, double> label_weights;

    // Calculate weights for each label based on distances
    for (int i = 0; i < k; ++i) {
        // Ensure we do not exceed the size of the sorted distances vector
        if (i < eucledian_distances.size()) {
            double distance = eucledian_distances[i].second;
            double weight = 1.0 / (distance + 1e-6); // Add a small epsilon to prevent division by zero
            std::string label = eucledian_distances[i].first;
            label_weights[label] += weight;
            weight_sum += weight;
        }
    }

    // Normalize weights
    for (auto& entry : label_weights) {
        entry.second /= weight_sum;
    }

    // Find the label with the highest weighted sum
    std::string best_label;
    double max_weight = 0.0;
    for (const auto& entry : label_weights) {
        if (entry.second > max_weight) {
            max_weight = entry.second;
            best_label = entry.first;
        }
    }

    return best_label;
}


/*
Description: Computes the closest label for a given feature vector by comparing it with the feature vectors in a database using scaled Euclidean distance.
Input: Takes a tuple containing the feature vectors of the object being detected (bounding box area and height-to-width ratio) and a vector of DataStruct objects representing the feature vectors in the database.
Output: Returns a string indicating the closest label to the detected object based on the scaled Euclidean distance comparison.
*/
std::string computeClosestLabel(const std::tuple<double, double, double, double>& feature_vectors, const std::vector<DataStruct>& feature_vector_data) {
    //Gets the bounding box area and the bounding box width to height ratio from the object being detected
    double live_bounding_box_area = std::get<0>(feature_vectors);
    double live_height_to_width_ratio = std::get<1>(feature_vectors);
    //setting the minimum distance to the maxium possible value to use this compare the other scaled eucledian distance values to filter the smallest value
    double min_distance = std::numeric_limits<double>::max();
    //std::string closest_label; // Used to store the closest label to the detected object

    // Calculate the standard deviatio of the feature vectors data inside the database(csv file)
    std::tuple<double, double> standardDeviation = computeFeatureStdDeviations(feature_vector_data);
    double standardDeviation_percentageFilled = std::get<0>(standardDeviation);//Standard deviation of data for percentage of area filled inside the bounding box
    double standardDeviation_heightToWidthRatio = std::get<1>(standardDeviation);//Standard deviation of the data for the height to width ration of the bounding box
    std::vector<std::pair<std::string, double>> distances;//Vector to store the scaled eucledian distance of the other features from the features of the object being detected
    // Iterate through each feature vector in the CSV data and calucalte the scaled eucledian distance
    //double min_distance = numeric_limits<double>::max(); // Initialize min_distance with a large value
    string closest_label = "Unknown Object"; // Default label assuming unknown object

    for (const auto& row : feature_vector_data) {
        // Compute the scaled Euclidean distance between the live feature vector and the CSV feature vector        
        double distance = scaledEuclideanDistance(row, live_height_to_width_ratio, live_bounding_box_area, standardDeviation_heightToWidthRatio, standardDeviation_percentageFilled);
        distances.push_back({ row.label ,distance });
        //cout << "Distances ::::: " << distance<<"Label is ::  "<<row.label << endl;
        // Update the minimum distance and closest label
        if (distance < min_distance) {
            min_distance = distance;
            cout << "Distance = " << distance << endl;
            if (min_distance < 0.1) {
                closest_label = row.label;
            }
        }
    }
    //waitKey(5000);
    //cout << "Closest Label: " << closest_label << endl;
    return closest_label;//returns the closest match
}

//**************Task-7*********************//


/*
 Description: This function computes features for regions in the segmented image and visualizes them on the original frame and draws the oriented bounding box.
 Input:
    - image_info: Information about the segmented regions including centroids and individual regions.
    - frame: The original frame where the regions and features will be visualized.
    - processed_image: The segmented image containing regions.
    - region_map: A map representing regions in the segmented image.
 Output:
    - Returns a tuple containing the percentage filled and height-to-width ratio of the regions.
    - The function also visualizes centroids, oriented bounding boxes, principal axes, and region numbers on the frame.
*/
tuple<double, double, double, double> find_feature_vectors_for_confusionmatrix(ImageInfo& image_info, Mat& frame, Mat& processed_image, Mat& region_map) {

    double height_to_width_ratio = 0;//Initilizing the height to width ratio to 0
    double percentage_filled = 0; //Initilizing the percentage filled to 0
    double centroid_x = 0;//Initilizing centroid value to 0
    double centroid_y = 0;
    int number_of_objects = 0;
    // Pre-allocate memory for contours vector    
    vector<vector<Point>> contours;
    contours.reserve(image_info.regiod_id.size());

    //Looping through image info to 
    for (int id : image_info.regiod_id) {
        if (id == 0) {
            continue;
        }

        // Retrieve centroid coordinates
        pair<double, double> centroid_coords = image_info.centroid[id];
        int x_int = static_cast<int>(centroid_coords.first);
        int y_int = static_cast<int>(centroid_coords.second);

        // Draw the centroid of the region
        cv::circle(frame, cv::Point(x_int, y_int), 5, cv::Scalar(0, 0, 255), -1);
        // Access the individual region corresponding to the given ID
        Mat individual_region = image_info.individual_region[id];
        number_of_objects++;
        // Find the contours of the individual region
        findContours(individual_region, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            // Compute the minimum area rectangle around the contours
            RotatedRect rotated_rect = minAreaRect(contours[0]);
            // Draw the oriented bounding box
            Point2f vertices[4];
            rotated_rect.points(vertices);
            for (int i = 0; i < 4; ++i) {
                line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(160, 32, 340), 2);
            }

            // Convert double coordinates to integers
            Point centroid(x_int, y_int);

            // Compute features for the region
            compute_features_for_region(frame, processed_image, id, centroid, region_map);
            centroid_x = centroid.x;
            centroid_y = centroid.y;

            // Calculate height-to-width ratio and percentage filled
            double bbox_width = rotated_rect.size.width;
            double bbox_height = rotated_rect.size.height;
            // Ensuring the height always remains the same irrespective of rotation
            if (bbox_height < bbox_width) {
                swap(bbox_height, bbox_width); // Swap the values if bbox_width is larger
            }
            height_to_width_ratio = bbox_height / bbox_width;//Calculating the bounding box height to width ratio
            double contour_area = contourArea(contours[0]);
            double bounding_box_area = bbox_width * bbox_height;
            percentage_filled = (contour_area / bounding_box_area) * 100.0;//Calculating the percentage filled in the bounding box

            // Convert the computed values to strings and draw on the image
            std::string ratio_text = "H/W Ratio: " + std::to_string(height_to_width_ratio);
            std::string percentage_text = "Filled: " + std::to_string(percentage_filled) + "%";
            cv::putText(frame, ratio_text, centroid, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);//Displaying the height to width ratio of the bounding box
            cv::putText(frame, percentage_text, cv::Point(centroid.x, centroid.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);//Displaying the percentage filled in the bounding box

        }
        // Clear contours vector for the next iteration
        contours.clear();
    }
    //cout << "Bounding box area : " << percentage_filled << endl;
    //cout << "Height to width ratio : " << height_to_width_ratio << endl;
    return make_tuple(percentage_filled, height_to_width_ratio, centroid_x, centroid_y);//Returning percentage fille, height to width ratio and the centroids values
}




// Define the Feedback struct to hold the actual and closest labels
struct Feedback {
    std::string actual_label;
    std::string closest_label;
};

Mat confusionMatrix = Mat::zeros(5, 5, CV_32SC1);//Initilizing the confusion matrix to zeros

/*
Description: Maps a class label to an index in the confusion matrix based on predefined label mappings.
Input: Takes a string representing the class label.
Output: Returns an integer index corresponding to the class label in the confusion matrix.
*/
int labelToIndex(const std::string& label) {
    // Map labels to indices based on the given labels in the CSV file
    if (label == "controller") {
        return 0;
    }
    else if (label == "opener") {
        return 1;
    }
    else if (label == "marker") {
        return 2;
    }
    else if (label == "calculator") {
        return 3;
    }
    else if (label == "earphones") {
        return 4;
    }
    else {
        // Handle unknown labels (if any)
        return -1; // or any other value indicating an error
    }
}


/*
Description: Updates the confusion matrix based on user feedback provided by comparing the actual label with the closest classified label.
Input: Takes a confusion matrix (cv::Mat), the actual label (string), the closest classified label (string), and a boolean flag indicating whether the classification was correct.
Output: Modifies the confusion matrix based on the provided feedback.
*/
void updateConfusionMatrix(cv::Mat& confusionMatrix, const std::string& actual_label, const std::string& closest_label, bool correctClassification) {
    int actualLabelIndex = labelToIndex(actual_label); // Convert true label to index in confusion matrix
    int closestLabelIndex = labelToIndex(closest_label); // Convert classified label to index in confusion matrix

    if (actualLabelIndex >= 0 && closestLabelIndex >= 0) {
        if (correctClassification) {
            // If classification is correct, increment the corresponding cell in the confusion matrix
            confusionMatrix.at<int>(actualLabelIndex, actualLabelIndex)++;
        }
        else {
            // If classification is incorrect, increment the cell corresponding to true label but classified incorrectly
            confusionMatrix.at<int>(actualLabelIndex, closestLabelIndex)++;
        }
    }
    else {
        // Handle invalid label indices
        std::cerr << "Invalid label indices." << std::endl;
    }
}







//Main code block
int main(int, char*) {

    Mat frame; //Initilizing a the frame
    Mat dilated_image;
    Mat binary_image;
    Mat individual_segmented_image;
    VideoCapture cap; //Creating a object called cap
    cap.open(1); //Opening the camera, I am using 1 for phone camera and uisng 0 for the laptops internal webcam
    vector<RegionFeature> features;//vector of type RegionFeatures(which is a structure)

    bool q_key_pressed = false;
    bool d_key_pressed = false;
    bool n_key_pressed = false;
    bool m_key_pressed = false;
    bool k_key_pressed = false;

    for (;;)
    {

        cap >> frame;
        Mat processed_image = preProcess_image(frame, 120, dilated_image, binary_image);//Processing the image

        //Performing connected component analysis

        std::tuple<ImageInfo, cv::Mat, cv::Mat> result = segment_the_image_into_regions(processed_image, individual_segmented_image); //Segment the image into regions, passing the eroded image to compute the info about the image using connectedcomponentswithstats
        ImageInfo image_info = get<0>(result);
        Mat region_stats = get<1>(result);
        Mat region_map = get<2>(result);
        vector<int> region_ids_to_draw;//vector to store the regions ids to draw
        vector<RegionFeature> region_features; // Define a vector to store computed features 



        // Handle key events
        char key = waitKey(1);

        if (key == 'q' && !q_key_pressed) {
            q_key_pressed = true;
            break; // Quit the loop if 'q' is pressed
        }


        for (int id : image_info.regiod_id) {
            if (id == 0) {
                continue;
            }

            tuple<double, double, double, double> feature_vectors = find_feature_vectors_id(image_info, frame, processed_image, region_map, id);
            double bounding_box_area = get<0>(feature_vectors);
            double height_to_width_ratio = get<1>(feature_vectors);
        }


        /*
        //This will compute the feature vectors such as "Height-to-width ratio" and the "Percentage filled"
        tuple<double, double, double, double> feature_vectors = find_feature_vectors(image_info, frame, processed_image, region_map);
        double bounding_box_area = get<0>(feature_vectors);
        double height_to_width_ratio = get<1>(feature_vectors);
        //cout << "Bounding box area : " << bounding_box_area << endl;
        //cout << "Height to width rotio : " << height_to_width_ratio << endl;
        */

        // When 'n' is pressed
        //When the user types an N, prompts the user for a label and then store the feature vector for the current object along with its label into a file
        if (key == 'n') {
            for (int id : image_info.regiod_id) {
                if (id == 0) {
                    continue;
                }
                // Set the label according to the user input
                string label;
                cout << "Enter the name of the object with id " + std::to_string(id) + " :";
                cin >> label;

                // Compute the feature vectors
                std::tuple<double, double, double, double> feature_vectors = find_feature_vectors_id(image_info, frame, processed_image, region_map, id);
                double bounding_box_area = get<0>(feature_vectors);
                double height_to_width_ratio = get<1>(feature_vectors);
                double centroid_x = get<2>(feature_vectors);
                double centroid_y = get<3>(feature_vectors);


                // Create a RegionFeature object to store the computed features and label
                RegionFeature region_feature;
                region_feature.label = label;
                region_feature.percentage_filled = bounding_box_area;
                region_feature.height_to_width_ratio = height_to_width_ratio;

                // Add the RegionFeature object to the vector of region features
                region_features.push_back(region_feature);
                save_to_csv(region_features);
                // Save the image with the label name
                string binary_filename = "D:\\MS Robotics Neu\\Spring 2024\\binary_" + label + ".png";
                string dilated_filename = "D:\\MS Robotics Neu\\Spring 2024\\dilated_" + label + ".png";
                string eroded_filename = "D:\\MS Robotics Neu\\Spring 2024\\eroded_" + label + ".png";
                string segmented_filename = "D:\\MS Robotics Neu\\Spring 2024\\segmented_" + label + ".png";
                string final_filename = "D:\\MS Robotics Neu\\Spring 2024\\final_" + label + ".png";
                cv::imwrite(binary_filename, binary_image);
                cv::imwrite(dilated_filename, dilated_image);
                cv::imwrite(eroded_filename, processed_image);
                cv::imwrite(segmented_filename, individual_segmented_image);
                cv::imwrite(final_filename, frame);

            }
            n_key_pressed = true; // Set the flag to true
        }

        else if (key == 'd' && !d_key_pressed) {
            d_key_pressed = true;
            std::vector<DataStruct> feature_vector_data = read_csv("D:\\MS Robotics Neu\\Spring 2024\\object_features.csv");

            for (int id : image_info.regiod_id) {
                if (id == 0) {
                    continue; // Skip the background region
                }

                // Get the feature vectors for the current object
                std::tuple<double, double, double, double> feature_vectors_live = find_feature_vectors_id(image_info, frame, processed_image, region_map, id);
                double centroid_x = get<2>(feature_vectors_live);
                double centroid_y = get<3>(feature_vectors_live);


                // Compute the closest label for the current object
                std::string closest_label = computeClosestLabel(feature_vectors_live, feature_vector_data);
                cout << "Closes Label : " << closest_label << endl;
                cout << "Centroids of object : " << id << "centroid_x : " << centroid_x << " centroid_y : " << centroid_y << endl;
                //waitKey(2000);
                // Display the assigned label on the frame and save it 
                cv::putText(frame, closest_label, Point(centroid_x + 30, centroid_y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);



                /*string detected_filename = "D:\\MS Robotics Neu\\Spring 2024\\Detected_" + closest_label + ".png";
                imwrite(detected_filename, frame);*/
            }
            string detected_filename = "D:\\MS Robotics Neu\\Spring 2024\\Detected_Multiple.png";
            cv::imwrite(detected_filename, frame);
        }

        //When m is pressed the procedure to compute the confusion matrix starts
        if (key == 'm') {
            string actual_label;
            cout << "Enter true label of the object: ";
            cin >> actual_label;

            std::tuple<double, double, double, double> feature_vectors = find_feature_vectors_for_confusionmatrix(image_info, frame, processed_image, region_map);//Computing the feature vectors
            std::vector<DataStruct> feature_vector_data = read_csv("D:\\MS Robotics Neu\\Spring 2024\\object_features.csv");//read the feature vectors from a csv file and store them in a datastructure

            std::string closest_label = computeClosestLabel(feature_vectors, feature_vector_data);//Computing the closest label
            bool TrueClassifier = (actual_label == closest_label);//clssifying

            int sample_size = 15;//Defining the sample size for calcualting the confustion matrix
            std::vector<Feedback> feedbacks;//Initilizing a vector called 'feedbacks' storing the user feedback

            for (int val = 0; val < sample_size; val++) {
                Feedback feedback;
                feedback.actual_label = actual_label;
                feedback.closest_label = closest_label;
                feedbacks.push_back(feedback);
            }
            // Print user feedbacks
            std::cout << "User Feedbacks:" << std::endl;
            for (const auto& feedback : feedbacks) {
                std::cout << "True Label: " << feedback.actual_label << ", Classified Label: " << feedback.closest_label << std::endl;
            }

            // Update confusion matrix based on user feedback
            updateConfusionMatrix(confusionMatrix, actual_label, closest_label, TrueClassifier);
            std::cout << "Confusion Matrix:" << std::endl << confusionMatrix << std::endl;

        }


        // When 'k' is pressed, detected object is calssified through KNN Matching
        if (key == 'k') {
            //Get the live data feature vectors from the image
            
            //Loading the training data from dataset into the vector of structures containig the label and the other 2 feature vectors
            std::vector<DataStruct> feature_vector_data = read_csv("D:\\MS Robotics Neu\\Spring 2024\\object_features.csv");
            for (int id : image_info.regiod_id) {
                if (id == 0) {
                    continue; // Skip the background region
                }


                // Get the feature vectors for the current object
                std::tuple<double, double, double, double> feature_vectors_live = find_feature_vectors_id(image_info, frame, processed_image, region_map, id);
                double live_bounding_box_area = get<0>(feature_vectors_live);
                double live_height_to_width_ratio = get<1>(feature_vectors_live);
                double centroid_x = get<2>(feature_vectors_live);
                double centroid_y = get<3>(feature_vectors_live);


                // Compute the closest label for the current object
                std::string closest_label = computeClosestLabel(feature_vectors_live, feature_vector_data);

                double min_distance = std::numeric_limits<double>::max();
                vector<pair<string, double>> eucledian_distances;
                // Calculate standard deviation outside the loop
                tuple<double, double> standardDeviation = computeFeatureStdDeviations(feature_vector_data);
                double standardDeviation_percentageFilled = get<0>(standardDeviation);
                double standardDeviation_heightToWidthRatio = get<1>(standardDeviation);
                // Calculate the Euclidean distance between the live image feature vectors and the image vectors from the database
                for (const auto& row : feature_vector_data) {
                    // Calculate the squared Euclidean distance without using pow
                    double dx = live_bounding_box_area - row.percentage_filed;
                    double dy = live_height_to_width_ratio - row.height_to_width_ratio;
                    double squared_distance = dx * dx + dy * dy;

                    // Calculate the Euclidean distance by taking the square root of the squared distance
                    double distance = sqrt(squared_distance);

                    // Store the distance along with the label
                    eucledian_distances.push_back({ row.label, distance });
                }

                // Sort the Euclidean distances in ascending order based on the distance values
                std::sort(eucledian_distances.begin(), eucledian_distances.end(),
                    [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                        return a.second < b.second; // Sort based on the distance (second element of the pair)
                    });

                int k = 3; // Number of nearest neighbors to consider

                // Classify using weighted KNN
                std::string classifiedLabel = knn_matching(eucledian_distances, k);
                std::cout << "Classified Label: " << classifiedLabel << std::endl;
                // Display the assigned label on the frame
                cv::putText(frame, classifiedLabel, Point(centroid_x, centroid_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                //string detected_KNN_Matching_filename = "D:\\MS Robotics Neu\\Spring 2024\\Detected_" + classifiedLabel + ".png";//Path to save the image
                //imwrite(detected_KNN_Matching_filename, frame);//Storing the detected image

            }
            string detected_KNN_Matching_filename = "D:\\MS Robotics Neu\\Spring 2024\\Detected_Detected_KNN.png";//Path to save the image
            cv::imwrite(detected_KNN_Matching_filename, frame);//Storing the detected image

        }


        //Displaying the thresholded image, dilated image, eroded image, segmented image and the final image
        //imshow("Binary Image", binary_image);
        //imshow("Dilated Image", dilated_image);
        //imshow("Eroded Image", processed_image);
        //imshow("Segmented Image", individual_segmented_image);
        imshow("Final Image", frame);


        // Check for key release events to enable toggling again
        if (key == 'q') {
            q_key_pressed = false;
        }
        else if (key == 'd') {
            d_key_pressed = false;
        }
        else if (key == 'n') {
            n_key_pressed = false;
        }
        else if (key == 'm') {
            m_key_pressed = false;
        }
        else if (key == 'k') {
            k_key_pressed = false;
        }

    }



    // Release resources
    cap.release(); // Release camera

    return 0;
}