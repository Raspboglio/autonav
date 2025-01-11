#include <FreeImage.h>
#include <colmap/controllers/feature_extraction.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <stereovision/CameraCalibrator.hpp>
#include <vector>

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
  mLeftKeypoints.resize(mStereoImages.size());
  mRightKeypoints.resize(mStereoImages.size());
  mLeftDescriptors.resize(mStereoImages.size());
  mRightDescriptors.resize(mStereoImages.size());
  mMatches.resize(mStereoImages.size());

  for (size_t i = 0; i < mStereoImages.size(); i++) {
    colmap::ImageData leftImageData, rightImageData;
    cv::Mat leftImage = mStereoImages[i]->getLeft();
    cv::Mat rightImage = mStereoImages[i]->getRight();

    leftImageData.bitmap = colmap::Bitmap::ConvertFromRawBits(
        leftImage.data, leftImage.cols * 3, leftImage.cols, leftImage.rows);
    leftImageData.bitmap = leftImageData.bitmap.CloneAsGrey();
    rightImageData.bitmap = colmap::Bitmap::ConvertFromRawBits(
        rightImage.data, rightImage.cols * 3, rightImage.cols, rightImage.rows);
    rightImageData.bitmap = rightImageData.bitmap.CloneAsGrey();
    // For each image, resize and extract the features

    // Resize

    // Extract

    mLeftKeypoints[i] = std::make_shared<colmap::FeatureKeypoints>();
    mRightKeypoints[i] = std::make_shared<colmap::FeatureKeypoints>();
    mLeftDescriptors[i] = std::make_shared<colmap::FeatureDescriptors>();
    mRightDescriptors[i] = std::make_shared<colmap::FeatureDescriptors>();

    std::cout << "[CameraCalibrator] Extracting features" << std::endl;

    mExtractor->Extract(leftImageData.bitmap, mLeftKeypoints[i].get(),
                        mLeftDescriptors[i].get());
    mExtractor->Extract(rightImageData.bitmap, mRightKeypoints[i].get(),
                        mRightDescriptors[i].get());

    // DEBUG PLOTS
    // std::cout << "Left Keypoints:\n";
    // for(const auto& keypoint: mLeftKeypoints[i]){
    //   std::cout  << keypoint.x << " " << keypoint.y << std::endl;
    //   cv::Point center(keypoint.x, keypoint.y);
    //   cv::circle(leftImage, center, keypoint.ComputeScale(), cv::Scalar(0, 0,
    //   255), 2);
    // }
    // cv::imwrite("left_image_keypoints.png", leftImage);

    // Match
    colmap::FeatureMatcher::Image leftMatchImage, rightMatchImage;
    leftMatchImage.image_id = i;
    leftMatchImage.descriptors = mLeftDescriptors[i];
    leftMatchImage.keypoints = mLeftKeypoints[i];
    rightMatchImage.image_id = i;
    rightMatchImage.descriptors = mRightDescriptors[i];
    rightMatchImage.keypoints = mRightKeypoints[i];
    mMatches[i] = std::make_shared<colmap::FeatureMatches>();

    std::cout << "[CameraCalibrator] Matching" << std::endl;
    mMatcher->Match(leftMatchImage, rightMatchImage, mMatches[i].get());
    // DEBUG PLOT
    // std::cout << "[CameraCalibrator] Concat" << std::endl;
    // std::vector<cv::Mat> images;
    // images.push_back(leftImage);
    // images.push_back(rightImage);
    // cv::Mat imageConcat;
    // cv::hconcat(images, imageConcat);
    // cv::imwrite("concat.png", imageConcat);

    // std::cout << "[CameraCalibrator] Drawing " << matches->size() << "
    // Matches"
    //           << std::endl;
    // cv::RNG rng(12345);
    // for (const auto& match : *matches) {
    //   colmap::FeatureKeypoint leftkey =
    //   mLeftKeypoints[i][match.point2D_idx1]; colmap::FeatureKeypoint rightkey
    //   = mRightKeypoints[i][match.point2D_idx2]; cv::Point
    //   leftPoint(leftkey.x, leftkey.y); cv::Point rightPoint(rightkey.x +
    //   leftImage.cols, rightkey.y); std::cout << "[CameraCalibrator]
    //   KeypointLeft: " << leftPoint.x << " "
    //             << leftPoint.y << "\nKeypointRight: " << rightPoint.x << " "
    //             << rightPoint.y << std::endl;

    //   cv::Scalar color =
    //       cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0,
    //       255));

    //   cv::circle(imageConcat, leftPoint, leftkey.ComputeScale(), color, 2);
    //   cv::circle(imageConcat, rightPoint, leftkey.ComputeScale(), color, 2);
    //   cv::line(imageConcat, leftPoint, rightPoint, color);
    // }
    // cv::imwrite("matches.png", imageConcat);
    // std::cout << "[CameraCalibrator] Finished" << std::endl;
  }

  return true;
}

void CameraCalibrator::printFeatures(std::string aPath) {
  for (size_t i = 0; i < mStereoImages.size(); i++) {
    cv::Mat imageConcat;
    std::vector<cv::Mat> images = {mStereoImages[i]->getLeft(),
                                   mStereoImages[i]->getRight()};
    cv::hconcat(images, imageConcat);

    cv::RNG rng(12345);
    for (const auto& match : *mMatches[i]) {
      colmap::FeatureKeypoint leftkey =
          (*mLeftKeypoints[i])[match.point2D_idx1];
      colmap::FeatureKeypoint rightkey =
          (*mRightKeypoints[i])[match.point2D_idx2];
      cv::Point leftPoint(leftkey.x, leftkey.y);
      cv::Point rightPoint(rightkey.x + mStereoImages[i]->getLeft().cols,
                           rightkey.y);
      cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                    rng.uniform(0, 255));
      cv::circle(imageConcat, leftPoint, leftkey.ComputeScale(), color, 2);
      cv::circle(imageConcat, rightPoint, leftkey.ComputeScale(), color, 2);
      cv::line(imageConcat, leftPoint, rightPoint, color);
    }
    cv::imwrite(aPath + "/matches" + std::to_string(i) + ".png", imageConcat);
  }
}

}  // namespace autonav::stereovision