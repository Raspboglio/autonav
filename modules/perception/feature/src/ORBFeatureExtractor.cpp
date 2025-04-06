#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Types.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>

#include <cmath>
#include <cstddef>
#include <feature/ORBFeatureExtractor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vpi/OpenCVInterop.hpp>

namespace autonav::feature {
bool ORBFeatureExtractor::extractFeatures(const cv::Mat& aImage,
                                          std::vector<cv::KeyPoint>& aKeypoints,
                                          std::vector<cv::Mat>& aDescriptors) {

  // We now wrap the loaded image into a VPIImage object to be used by VPI.
  // VPI won't make a copy of it, so the original image must be in scope at
  VPIStatus res;

  res = vpiImageCreateWrapperOpenCVMat(aImage, 0, &mImage);
  if(!checkRes(res)){
    std::cout << "[ORBFeatureExtractor::extractFeatures] Error wrapping CV image\n";
    return false;
  }
  res = vpiImageCreate(aImage.cols, aImage.rows, VPI_IMAGE_FORMAT_U8, 0, &mImageGray);
  if(!checkRes(res)){
    std::cout << "[ORBFeatureExtractor::extractFeatures] Error creating image\n";
    return false;
  }
  // For the output arrays capacity we can use the maximum number of
  // features per level multiplied by the maximum number of pyramid
  // levels, this will be the de factor maximum for all levels of the
  // input.
  // Create the output keypoint array.

  // For the internal buffers capacity we can use the maximum number of
  // features per level multiplied by 20. This will make FAST find a large
  // number of corners so then ORB can select the top N corners in accordance
  // to Harris score of each corner, where N = maximum number of features per
  // level.

  // ================
  // Processing stage

  // First convert input to grayscale
  res = vpiSubmitConvertImageFormat(mStream, mBackend, mImage, mImageGray, NULL);
  if(!checkRes(res)){
    std::cout << "[ORBFeatureExtractor::extractFeatures] ERROR Converting image to Grayscale\n";
    return false;
  }
  // Then, create the Gaussian Pyramid for the image and wait for the
  // execution to finish

  res = vpiPyramidCreate(aImage.cols, aImage.rows, VPI_IMAGE_FORMAT_U8,
                   mOrbParams.maxPyramidLevels, 0.5, mBackend, &mPyrInput);
  if(!checkRes(res)){
    std::cout << "[ORBFeatureExtractor::extractFeatures] ERROR creating VPI pyramid\n";
    return false;
  }

  res = vpiSubmitGaussianPyramidGenerator(mStream, mBackend, mImageGray, mPyrInput,
                                    VPI_BORDER_CLAMP);
  if(!checkRes(res)){
    std::cout << "[ORBFeatureExtractor::extractFeatures] ERROR generating pyramid\n";
    return false;
  }

  // Then get ORB features and wait for the execution to finish
  res = vpiSubmitORBFeatureDetector(mStream, mBackend, mExtractor, mPyrInput,
                              mKeypoints, mDescriptors, &mOrbParams,
                              VPI_BORDER_LIMITED);
  if(!checkRes(res)){
    std::cout << "[ORBFeatureExtractor::extractFeatures] ERROR starting execution\n";
    return false;
  }

  vpiStreamSync(mStream);

  // =======================================
  // Output processing and saving it to disk

  // Lock output keypoints and scores to retrieve its data on cpu memory
  VPIArrayData outKeypointsData;
  VPIArrayData outDescriptorsData;
  VPIImageData imgData;

  vpiArrayLockData(mKeypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                   &outKeypointsData);
  vpiArrayLockData(mDescriptors, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                   &outDescriptorsData);
  vpiImageLockData(mImageGray, VPI_LOCK_READ,
                   VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData);

  VPIPyramidalKeypointF32* outKeypoints =
      (VPIPyramidalKeypointF32*)outKeypointsData.buffer.aos.data;
  VPIBriefDescriptor* outDescriptors =
      (VPIBriefDescriptor*)outDescriptorsData.buffer.aos.data;

  // Unlock resources
  vpiImageUnlock(mImageGray);
  vpiArrayUnlock(mKeypoints);
  vpiArrayUnlock(mDescriptors);

  aKeypoints.resize(*outKeypointsData.buffer.aos.sizePointer);
  for (size_t i = 0; i < *outKeypointsData.buffer.aos.sizePointer; i++) {
    float scale = std::pow(2,outKeypoints[i].octave);
    aKeypoints[i].pt.x = outKeypoints[i].x * scale;
    aKeypoints[i].pt.y = outKeypoints[i].y * scale;
    aKeypoints[i].octave = outKeypoints[i].octave;
    aKeypoints[i].angle = -1;
    aKeypoints[i].size = 100.0 / scale;
  }

  aDescriptors.resize(*outKeypointsData.buffer.aos.sizePointer);
  for (size_t i = 0; i < outDescriptorsData.buffer.aos.capacity; i++) {
    aDescriptors[i].data = outDescriptors[i].data;
    aDescriptors[i].dataend = outDescriptors[i].data + 32;
    aDescriptors[i].datastart = outDescriptors[i].data;
  }
  return true;
}

bool ORBFeatureExtractor::checkRes(VPIStatus& res) {
  if (res != VPI_SUCCESS) {
    char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(buffer, sizeof(buffer));
    std::cout << "Error: " << buffer << "\n";
  }
  return res == VPI_SUCCESS;
}

}  // namespace autonav::feature
