#ifndef CAMERA_CALIBRATOR_HPP
#define CAMERA_CALIBRATOR_HPP
#include <colmap/feature/extractor.h>
#include <colmap/feature/matcher.h>
#include <colmap/feature/sift.h>

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

 private:
  std::unique_ptr<colmap::FeatureExtractor> mExtractor;
  std::unique_ptr<colmap::FeatureMatcher> mMatcher;

  std::vector<autonav::utils::StereoImage::UniquePtr> mStereoImages;
};

}  // namespace autonav::stereovision

#endif  // CAMERA_CALIBRATOR_HPP