#include <feature/ORBFeatureExtractor.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <stereovision/CameraCalibrator.hpp>
#include <utils/StereoImage.hpp>

#include "colmap/feature/sift.h"

#define MAX_IMAGES 1000

int main(int argc, char** argv) {
  std::string resultPath, datasetPath;
  if (argc == 2) {
    std::cout << "[cameraExtractionTest]: Using current folder as result_folder"
              << std::endl;
    resultPath = ".";
  } else if (argc == 3) {
    std::cout << "[cameraExtractionTest]: Using " << argv[2]
              << " as result_folder" << std::endl;
    resultPath = argv[2];
  } else {
    std::cerr
        << "Usage: " << argv[0]
        << " <dataset_folder> <result_folder>(optional) \n\n are expected in "
           "the following format\n\tcamera<[0,1]><camera_id>.png\n\t\t 0 is "
           "for left images and 1 for right images"
        << std::endl;
    return -1;
  }
  datasetPath = argv[1];

  colmap::SiftExtractionOptions extOptions;
  extOptions.num_threads = 1;
  extOptions.use_gpu = true;

  colmap::SiftMatchingOptions matchOptions;
  matchOptions.use_gpu = true;
  matchOptions.max_num_matches = 10000;

  autonav::stereovision::CameraCalibrator calibrator(extOptions, matchOptions);

  bool imageFound = true;
  size_t imageIndex = 1;

  while (imageFound && imageIndex < MAX_IMAGES) {
    cv::Mat left, right;
    left = cv::imread(
        datasetPath + "/camera" + std::to_string(imageIndex) + "0.png",
        cv::IMREAD_COLOR);
    if (left.data == NULL) {
      imageFound = false;
      std::cout << "[cameraExtractionTest]: Read up to index " << imageIndex - 1
                << std::endl;
      break;
    }
    right = cv::imread(
        datasetPath + "/camera" + std::to_string(imageIndex) + "1.png",
        cv::IMREAD_COLOR);
    if (right.data == NULL) {
      imageFound = false;
      std::cout << "[cameraExtractionTest]: Read up to index " << imageIndex - 1
                << std::endl;
      break;
    }
    autonav::utils::StereoImage::UniquePtr image =
        std::make_unique<autonav::utils::StereoImage>(left, right);
    calibrator.addStereoImage(std::move(image));

    imageIndex++;
  }
  if (imageIndex == MAX_IMAGES) {
    std::cout << "[CameraExtractionTest]: Max images reached" << std::endl;
  }
  // calibrator.calibrate();
  // calibrator.printFeatures(resultPath);

  std::cout << "[CameraExtractionTest]: Reading image" << std::endl;
  cv::Mat orbImage =
      cv::imread(datasetPath + "/camera10.png", cv::IMREAD_COLOR);

  autonav::feature::ORBFeatureExtractorParams::Ptr params =
      std::make_shared<autonav::feature::ORBFeatureExtractorParams>();
  params->backendCuda = true;
  autonav::feature::ORBFeatureExtractor extractor(params);

  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Mat> descriptors;
  std::cout << "[CameraExtractionTest]: Extracting features" << std::endl;
  extractor.extractFeatures(orbImage.clone(), keypoints, descriptors);

  cv::Mat featureImage = orbImage.clone();
  std::cout << "[CameraExtractionTest]: Drawing keypoints: " << keypoints.size() << std::endl;
  for(const auto keypoint : keypoints) {
    std::cout << "\tx,y: " << keypoint.pt.x << ", " << keypoint.pt.y << std::endl;
    std::cout << "\tsize: " << keypoint.size << std::endl;
    std::cout << "\tscale: " << keypoint.octave << std::endl;
  }
  cv::drawKeypoints(orbImage, keypoints, featureImage);
  cv::imwrite(resultPath + "/features.png", featureImage);
  return 0;
}