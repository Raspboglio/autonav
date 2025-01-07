#include "colmap/feature/sift.h"
#include <opencv2/imgcodecs.hpp>
#include <stereovision/CameraCalibrator.hpp>
#include <utils/StereoImage.hpp>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_left> <path_right>" << std::endl;
        return -1;
    }
    std::string path_left = argv[1], path_right = argv[2];
    cv::Mat left, right;
    left = cv::imread(path_left, cv::IMREAD_COLOR);
    right = cv::imread(path_right, cv::IMREAD_COLOR);
    autonav::utils::StereoImage::UniquePtr image = std::make_unique<autonav::utils::StereoImage>(left, right);

    colmap::SiftExtractionOptions extOptions;
    extOptions.num_threads = 1;
    extOptions.use_gpu = true;

    colmap::SiftMatchingOptions matchOptions;
    matchOptions.use_gpu = true;
    matchOptions.max_num_matches = 10000;

    autonav::stereovision::CameraCalibrator calibrator(extOptions, matchOptions);

    calibrator.addStereoImage(std::move(image));

    calibrator.calibrate();
    return 0;
}