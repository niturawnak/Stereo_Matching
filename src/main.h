#pragma once

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities);

void StereoEstimation_DP(
  const int& window_size_dynamic,
  const int& dmin,
  int height,
  int width,
  int weight,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& dynamic_disparities);

void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length);

int Dissimilarity(
  cv::Mat& image1, cv::Mat& image2,
  int half_window_size, int y_0, int i, int j);