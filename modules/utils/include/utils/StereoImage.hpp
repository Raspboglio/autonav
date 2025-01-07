
#ifndef STEREO_IMAGE_HPP
#define STEREO_IMAGE_HPP
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

namespace autonav::utils {

class StereoImage {
 public:
  using Ptr = std::shared_ptr<StereoImage>;
  using ConstPtr = std::shared_ptr<const StereoImage>;
  using UniquePtr = std::unique_ptr<StereoImage>;

  StereoImage(cv::Mat aLeft, cv::Mat aRight) : mLeft(aLeft), mRight(aRight) {
    cv::cvtColor(mLeft, mLeftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(mRight, mRightGray, cv::COLOR_BGR2GRAY);
  }
  StereoImage() {}

  const cv::Mat& getLeft() { return mLeft; }
  const cv::Mat& getRight() { return mRight; }
  const cv::Mat& getLeftGray() { return mLeftGray; }
  const cv::Mat& getRightGray() { return mRightGray; }

  /*!
   * @brief set the left image
   * @param aLeft Left image as cv::Mat object with BGR coding
   */
  void setLeft(const cv::Mat aLeft) {
    mLeft = aLeft;
    cv::cvtColor(mLeft, mLeftGray, cv::COLOR_BGR2GRAY);
  }

  /*!
   * @brief set the right image
   * @param aLeft Right image as cv::Mat object with BGR coding
   */
  void setRight(const cv::Mat aRight) {
    mRight = aRight;
    cv::cvtColor(mRight, mRightGray, cv::COLOR_BGR2GRAY);
  }

  /*!
   * @brief Save the images
   * @param aPath folder path to save the images, if empty they will be saved in
   * the current folder
   */
  void save(const std::string aPath = "") const {
    auto current_time = std::chrono::system_clock::now();
    if (aPath.empty()) {
      cv::imwrite("left-" +
                      std::to_string(current_time.time_since_epoch().count()) +
                      ".png",
                  mLeft);
      cv::imwrite("right-" +
                      std::to_string(current_time.time_since_epoch().count()) +
                      ".png",
                  mRight);

      cv::imwrite("left-bw-" +
                      std::to_string(current_time.time_since_epoch().count()) +
                      ".png",
                  mLeftGray);
      cv::imwrite("right-bw-" +
                      std::to_string(current_time.time_since_epoch().count()) +
                      ".png",
                  mRightGray);
    } else {
      cv::imwrite(aPath + "/left-" +
                      std::to_string(current_time.time_since_epoch().count()) +
                      ".png",
                  mLeft);
      cv::imwrite(aPath + "/right-" +
                      std::to_string(current_time.time_since_epoch().count()) +
                      ".png",
                  mRight);
    }
  }

 private:
  cv::Mat mLeft, mRight, mLeftGray, mRightGray;
};
}  // namespace autonav::utils

#endif  // STEREO_IMAGE_HPP