#include <FreeImage.h>
#include <colmap/controllers/feature_extraction.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <stereovision/CameraCalibrator.hpp>

#include "colmap/feature/matcher.h"
#include "colmap/feature/types.h"
#include "colmap/scene/image.h"
#include "colmap/sensor/bitmap.h"

namespace autonav::stereovision {

void CameraCalibrator::addStereoImage(
    autonav::utils::StereoImage::UniquePtr aStereoImage) {
  mStereoImages.emplace_back(std::move(aStereoImage));
}

bool CameraCalibrator::calibrate() {
  for (const auto& stereoImage : mStereoImages) {
    colmap::ImageData leftImageData, rightImageData;
    cv::Mat leftImage = stereoImage->getLeft();
    cv::Mat rightImage = stereoImage->getRight();

    leftImageData.bitmap = colmap::Bitmap::ConvertFromRawBits(
        leftImage.data, leftImage.cols * 3, leftImage.cols, leftImage.rows);
    leftImageData.bitmap = leftImageData.bitmap.CloneAsGrey();
    rightImageData.bitmap = colmap::Bitmap::ConvertFromRawBits(
        rightImage.data, rightImage.cols * 3, rightImage.cols, rightImage.rows);
    rightImageData.bitmap = rightImageData.bitmap.CloneAsGrey();
    // For each image, resize and extract the features

    // Resize

    // Extract
    std::shared_ptr<colmap::FeatureKeypoints> leftKeypoints, rightKeypoints;
    leftKeypoints = std::make_shared<colmap::FeatureKeypoints>();
    rightKeypoints = std::make_shared<colmap::FeatureKeypoints>();
    std::shared_ptr<colmap::FeatureDescriptors> leftDescriptors, rightDescriptors;
    leftDescriptors = std::make_shared<colmap::FeatureDescriptors>();
    rightDescriptors = std::make_shared<colmap::FeatureDescriptors>();
    
    std::cout << "[CameraCalibrator] Extracting features" << std::endl;

    mExtractor->Extract(leftImageData.bitmap, leftKeypoints.get(), leftDescriptors.get());
    mExtractor->Extract(rightImageData.bitmap, rightKeypoints.get(),
                        rightDescriptors.get());

    // DEBUG PLOTS
    // std::cout << "Left Keypoints:\n";
    // for(const auto& keypoint: leftKeypoints){
    //   std::cout  << keypoint.x << " " << keypoint.y << std::endl;
    //   cv::Point center(keypoint.x, keypoint.y);
    //   cv::circle(leftImage, center, keypoint.ComputeScale(), cv::Scalar(0, 0,
    //   255), 2);
    // }
    // cv::imwrite("left_image_keypoints.png", leftImage);

    // Match
    colmap::FeatureMatcher::Image leftMatchImage, rightMatchImage;
    leftMatchImage.image_id = 0;
    leftMatchImage.descriptors = leftDescriptors;
    leftMatchImage.keypoints = leftKeypoints;
    rightMatchImage.image_id = 0;
    rightMatchImage.descriptors = rightDescriptors;
    rightMatchImage.keypoints = rightKeypoints;
    colmap::FeatureMatches* matches = new colmap::FeatureMatches();

    std::cout << "[CameraCalibrator] Matching" << std::endl;
    mMatcher->Match(leftMatchImage, rightMatchImage, matches);

    // DEBUG PLOT
    // std::cout << "[CameraCalibrator] Concat" << std::endl;
    // std::vector<cv::Mat> images;
    // images.push_back(leftImage);
    // images.push_back(rightImage);
    // cv::Mat imageConcat;
    // cv::hconcat(images, imageConcat);
    // cv::imwrite("concat.png", imageConcat);

    // std::cout << "[CameraCalibrator] Drawing " << matches->size() << " Matches"
    //           << std::endl;
    // cv::RNG rng(12345);
    // for (const auto& match : *matches) {
    //   colmap::FeatureKeypoint leftkey = leftKeypoints[match.point2D_idx1];
    //   colmap::FeatureKeypoint rightkey = rightKeypoints[match.point2D_idx2];
    //   cv::Point leftPoint(leftkey.x, leftkey.y);
    //   cv::Point rightPoint(rightkey.x + leftImage.cols, rightkey.y);
    //   std::cout << "[CameraCalibrator] KeypointLeft: " << leftPoint.x << " "
    //             << leftPoint.y << "\nKeypointRight: " << rightPoint.x << " "
    //             << rightPoint.y << std::endl;

    //   cv::Scalar color =
    //       cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

    //   cv::circle(imageConcat, leftPoint, leftkey.ComputeScale(), color, 2);
    //   cv::circle(imageConcat, rightPoint, leftkey.ComputeScale(), color, 2);
    //   cv::line(imageConcat, leftPoint, rightPoint, color);
    // }
    // cv::imwrite("matches.png", imageConcat);
    // std::cout << "[CameraCalibrator] Finished" << std::endl;
  }

  return true;
}

}  // namespace autonav::stereovision