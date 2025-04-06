#ifndef ORB_FEATURE_EXTRACTOR_HPP
#define ORB_FEATURE_EXTRACTOR_HPP

#ifdef USE_VPI
#include <vpi/Status.h>
#include <vpi/Types.h>
#include <vpi/VPI.h>
#include <vpi/algo/ORB.h>
#endif

#include <feature/FeatureExtractor.hpp>
#include <memory>

//
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace autonav::feature {

class ORBFeatureExtractorParams : public FeatureExtractorParams {
 public:
  using Ptr = std::shared_ptr<ORBFeatureExtractorParams>;
  ORBFeatureExtractorParams() {
    backendCuda = false;
    intensityTh = 50.0;
    featurePerLevel = 88;
    pyramidLevel = 3;
  }
  ~ORBFeatureExtractorParams() {}

  bool backendCuda;
  float intensityTh;
  int32_t featurePerLevel;
  int32_t pyramidLevel;
};

class ORBFeatureExtractor : public FeatureExtractor {
 public:
  ORBFeatureExtractor(FeatureExtractorParams::Ptr aParams =
                          std::make_shared<ORBFeatureExtractorParams>())
      : mParams(aParams) {
    ORBFeatureExtractorParams::Ptr params;
    params = std::dynamic_pointer_cast<ORBFeatureExtractorParams>(mParams);

    VPIBackend backendCpu;
    if (params->backendCuda) {
      backendCpu = static_cast<VPIBackend>(VPI_BACKEND_CUDA | VPI_BACKEND_CPU);
      mBackend = VPI_BACKEND_CUDA;
    } else {
      mBackend = VPI_BACKEND_CPU;
    }

    VPIStatus res = vpiStreamCreate(0, &mStream);
    if (!checkRes(res)) {
      std::cout << "[ORBFeatureExtractor]: vpiStreamCreate" << std::endl;
      // TODO: do something
    }

    res = vpiInitORBParams(&mOrbParams);
    if (!checkRes(res)) {
      std::cout << "[ORBFeatureExtractor]: vpiInitORBParams" << std::endl;
      // TODO: do something
    }
    mOrbParams.fastParams.intensityThreshold = params->intensityTh;
    mOrbParams.maxFeaturesPerLevel = params->featurePerLevel;
    mOrbParams.maxPyramidLevels = params->pyramidLevel;

    int outCapacity =
        mOrbParams.maxFeaturesPerLevel * mOrbParams.maxPyramidLevels;
    int bufCapacity = mOrbParams.maxFeaturesPerLevel * 20;

    res = vpiArrayCreate(outCapacity, VPI_ARRAY_TYPE_PYRAMIDAL_KEYPOINT_F32,
                         backendCpu, &mKeypoints);
    if (!checkRes(res)) {
      std::cout << "[ORBFeatureExtractor]: vpiArrayCreate" << std::endl;
      // TODO: do something
    }
    res = vpiArrayCreate(outCapacity, VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR,
                         backendCpu, &mDescriptors);
    if (!checkRes(res)) {
      std::cout << "[ORBFeatureExtractor]: vpiArrayCreate" << std::endl;

      // TODO: do something
    }

    res = vpiCreateORBFeatureDetector(mBackend, bufCapacity, &mExtractor);
    if (!checkRes(res)) {
      std::cout << "[ORBFeatureExtractor]: vpiCreateORBFeatureDetector"
                << std::endl;
      // TODO: do something
    }
  }
  ~ORBFeatureExtractor() {}

  bool extractFeatures(const cv::Mat& aImage,
                       std::vector<cv::KeyPoint>& aKeypoints,
                       std::vector<cv::Mat>& aDescriptors) override;

  bool issueExtraction();

 private:
  FeatureExtractorParams::Ptr mParams;

  VPIBackend mBackend;

  VPIStream mStream;
  VPIPayload mExtractor;
  VPIORBParams mOrbParams;

  VPIImage mImage = {NULL};
  VPIImage mImageGray = {NULL};
  VPIPyramid mPyrInput = {NULL};

  VPIArray mKeypoints = {NULL};
  VPIArray mDescriptors = {NULL};

  bool checkRes(VPIStatus& res);

  ///
  static cv::Mat DrawKeypoints(cv::Mat img, VPIPyramidalKeypointF32* kpts,
                               VPIBriefDescriptor* descs, int numKeypoints) {
    cv::Mat out;
    img.convertTo(out, CV_8UC1);
    cvtColor(out, out, cv::COLOR_GRAY2BGR);

    if (numKeypoints == 0) {
      return out;
    }

    std::vector<int> distances(numKeypoints, 0);
    float maxDist = 0.f;

    for (int i = 0; i < numKeypoints; i++) {
      for (int j = 0; j < VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH; j++) {
        distances[i] += std::bitset<8 * sizeof(uint8_t)>(descs[i].data[j] ^
                                                         descs[0].data[j])
                            .count();
      }
      if (distances[i] > maxDist) {
        maxDist = distances[i];
      }
    }

    uint8_t ids[256];
    std::iota(&ids[0], &ids[0] + 256, 0);
    cv::Mat idsMat(256, 1, CV_8UC1, ids);

    cv::Mat cmap;
    applyColorMap(idsMat, cmap, cv::COLORMAP_JET);

    for (int i = 0; i < numKeypoints; i++) {
      int cmapIdx =
          static_cast<int>(std::round((distances[i] / maxDist) * 255));

      float rescale = std::pow(2, kpts[i].octave);
      float x = kpts[i].x * rescale;
      float y = kpts[i].y * rescale;

      std::cout << "Keypoint position : " << x << "," << y << "\n";

      circle(out, cv::Point(x, y), 3, cmap.at<cv::Vec3b>(cmapIdx, 0), -1);
    }

    return out;
  }
};

}  // namespace autonav::feature

#endif  // ORB_FEATURE_EXTRACTOR_HPP