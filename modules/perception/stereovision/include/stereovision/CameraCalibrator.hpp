#ifndef CAMERA_CALIBRATOR_HPP
#define CAMERA_CALIBRATOR_HPP
#include <colmap/feature/extractor.h>
#include <colmap/feature/matcher.h>
#include <colmap/feature/sift.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <utils/StereoImage.hpp>
#include <vector>

namespace autonav::stereovision {

class CameraCalibrator {
 public:
  CameraCalibrator(colmap::SiftExtractionOptions aExtractorOptions,
                   colmap::SiftMatchingOptions aMatcherOptions) {
    mExtractor = colmap::CreateSiftFeatureExtractor(aExtractorOptions);
    mMatcher = CreateSiftFeatureMatcher(aMatcherOptions);
  }

  ~CameraCalibrator() {}

  void addStereoImage(autonav::utils::StereoImage::UniquePtr aStereoImage);

  bool calibrate();

  /**!
   *@brief Print results in specified directory
   *@param aPath Directory to print results in, if none is specified, the
   *current directory is used.
   */
  void printFeatures(std::string aPath = ".");

 private:
  std::unique_ptr<colmap::FeatureExtractor> mExtractor;
  std::unique_ptr<colmap::FeatureMatcher> mMatcher;

  std::vector<std::shared_ptr<colmap::FeatureKeypoints>> mLeftKeypoints;
  std::vector<std::shared_ptr<colmap::FeatureKeypoints>> mRightKeypoints;
  std::vector<std::shared_ptr<colmap::FeatureDescriptors>> mLeftDescriptors;
  std::vector<std::shared_ptr<colmap::FeatureDescriptors>> mRightDescriptors;

  std::vector<std::shared_ptr<colmap::FeatureMatches>> mMatches;

  std::vector<autonav::utils::StereoImage::UniquePtr> mStereoImages;
};

}  // namespace autonav::stereovision

#endif  // CAMERA_CALIBRATOR_HPP