
#include <memory>
#include <opencv2/opencv.hpp>

namespace autonav::feature {

class FeatureExtractorParams {
 public:
  using Ptr = std::shared_ptr<FeatureExtractorParams>;
  FeatureExtractorParams() {}
  virtual ~FeatureExtractorParams(){};
};
class FeatureExtractor {
 public:
  FeatureExtractor() {}
  ~FeatureExtractor() {}
  virtual bool extractFeatures(const cv::Mat& image,
                               std::vector<cv::KeyPoint>& keypoints,
                               std::vector<cv::Mat>& descriptors) = 0;
};
}  // namespace autonav::feature