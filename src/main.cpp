#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "main.h"

int main(int argc, char** argv) {

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 160;

  // stereo estimation parameters
  const int dmin = 200; //for Art, Books, Dolls, Moebius
  //const int dmin = 230; //for laundry, Reindeer
  
  //const int window_size = 3;

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE WINDOW_SIZE WEIGHT" << std::endl;
    return 1;
  }

  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); //pixel format uchar.. 8bit 
  cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);//pixel format uchar.. 8bit 
  const std::string output_file = argv[3];
  const int window_size= atoi(argv[4]);
  const int weight = atoi(argv[5]);


  if (!image1.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }

  if (!image2.data) {
    std::cerr << "No image2 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size = " << window_size << std::endl;
  std::cout << "Occlusion weight = " << weight << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;

  //processing time
  std::stringstream outTime;
  outTime << output_file << "_processing_time.txt";
  std::ofstream outfileTime(outTime.str());

  ////////////////////
  // Reconstruction //
  ////////////////////

  // Naive disparity image
  //cv::Mat naive_disparities = cv::Mat::zeros(height - window_size, width - window_size, CV_8UC1);
  cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);
  
  double current_time;
  current_time = (double)cv::getTickCount();
  StereoEstimation_Naive(
    window_size, dmin, height, width,
    image1, image2,
    naive_disparities);

  current_time = ((double)cv::getTickCount() - current_time)/cv::getTickFrequency();
  outfileTime << "Naive: " << current_time << " seconds" << std::endl;

  ////////////
  // Output //
  ////////////

  // save / display images
  std::stringstream out1;
  out1 << output_file << "_naive.png";
  cv::imwrite(out1.str(), naive_disparities);

  cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
  cv::imshow("Naive", naive_disparities);
  
  //////////////////////////////////
  // Dynamic Programming Approach //
  /////////////////////////////////

  // dynamic disparity image
  cv::Mat dynamic_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  current_time = (double)cv::getTickCount();
  StereoEstimation_DP(
    window_size, dmin, height, width, weight,
    image1, image2,
    dynamic_disparities);
  current_time = ((double)cv::getTickCount() - current_time)/cv::getTickFrequency();
  outfileTime << "Dynamic Programming: " << current_time << " seconds" << std::endl;
  // save / display images
  std::stringstream out2;
  out2 << output_file << "_dynamic.png";

  cv::Mat dynamic_disparities_image;
  cv::normalize(dynamic_disparities, dynamic_disparities_image, 255, 0, cv::NORM_MINMAX);

  cv::imwrite(out2.str(), dynamic_disparities_image);

  cv::namedWindow("Dynamic", cv::WINDOW_AUTOSIZE);
  cv::imshow("Dynamic", dynamic_disparities_image);

  // reconstruction
  // Disparity2PointCloud(
  //   output_file,
  //   height, width, dynamic_disparities_image,
  //   window_size, dmin, baseline, focal_length);


  // OpenCV implementation

  current_time = (double)cv::getTickCount();
  cv::Mat opencv_disparities;
  cv::Ptr<cv::StereoBM > match = cv::StereoBM::create(16, 9);
  match->compute(image1, image2, opencv_disparities);
  current_time = ((double)cv::getTickCount() - current_time)/cv::getTickFrequency();
  outfileTime << "StereoBM: " << current_time << " seconds" << std::endl;

  cv::imshow("OpenCV result",opencv_disparities*1000);

  std::stringstream out3;
  out3 << output_file << "_opencv.png";
  cv::imwrite(out3.str(), opencv_disparities);
  cv::waitKey(0);

  return 0;
}

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
  int half_window_size = window_size / 2;
  int progress =0;
//OpenMP
#pragma omp parallel for //if 12 core , 12 threads in parallel pool

  for (int i = half_window_size; i < height - half_window_size; ++i) { //for each row

#pragma omp critical 
  {
    ++progress;
    std::cout
      << "Calculating disparities for the naive approach... "
      << std::ceil(((progress) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
      << std::flush;
  }

    for (int j = half_window_size; j < width - half_window_size; ++j) { //for each colum in the left image

    //(i,j) == (row, col)
      int min_ssd = INT_MAX;
      int disparity = 0;
    // for each (i,j)
    // do a 1D search for the best disparity
    // == minimum search SSD
      for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
        int ssd = 0;

        // TODO: sum up matching cost (ssd) in a window
        for (int m = -half_window_size; m <= half_window_size; ++m) { //row

          for (int n = -half_window_size; n <= half_window_size; ++n) { //col

            int left_img_val = image1.at<uchar>(i + m, j + n);
            int right_img_val = image2.at<uchar>(i + m, j + n + d);
            ssd += (left_img_val - right_img_val) * (left_img_val - right_img_val);

          }
        }

        if (ssd < min_ssd) {
          min_ssd = ssd;
          disparity = d;
        }
      }

      naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity);
    }
  }

  std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
  std::cout << std::endl;
}


void StereoEstimation_DP(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  int weight,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& dynamic_disparities)
{
  int half_window_size = window_size / 2;

  //for each row(scanline)
  for (int y_0 = half_window_size;y_0<height - half_window_size; ++y_0){
    std::cout
      << "Calculating disparities for the dynamic approach... "
      << std::ceil(((y_0 - half_window_size +1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
      << std::flush;

    //allocate C, M 
    cv::Mat C = cv::Mat::zeros(width, width, CV_32FC1);
    cv::Mat M = cv::Mat::zeros(width, width, CV_8UC1); //match 1, left-occlusion 2, right-occlusion 3

    //initialize C and M
    for (int x = 1; x < width; ++x){
      C.at<float>(x, 0) = x * weight;
      M.at<uchar>(x, 0) = 3;
    }

    for (int y = 1; y < width; ++y){
      C.at<float>(0, y) = y * weight;
      M.at<uchar>(0, y) = 2;
    }

    for (int i = 1; i < width; ++i){ //left image
      for (int j = 1; j < width; ++j){ //right image

        double dissim = Dissimilarity(image1, image2, half_window_size, y_0, i, j);
        double match_cost = C.at<float>(i-1, j-1) + dissim;
        double left_occl_cost = C.at<float>(i-1, j) + weight;
        double right_occl_cost = C.at<float>(i, j-1) + weight;

        if (match_cost < std::min(left_occl_cost, right_occl_cost)){
          C.at<float>(i, j) = match_cost;
          M.at<uchar>(i, j) = 1;
        }
        else if (left_occl_cost < std::min(match_cost, right_occl_cost)){
          C.at<float>(i, j) = left_occl_cost;
          M.at<uchar>(i, j) = 2;
        }
        else{ // (right_occl_cost < std::min(match_cost, left_occl_cost))
          C.at<float>(i, j) = right_occl_cost;
          M.at<uchar>(i, j) = 3;
        }

      }
    }

    // trace sink->source (from bottom-right to top-left of C/M)
    int x = width - 1;
    int y = width - 1;
    int d = 0;
    while (x != 0 && y != 0){
      switch (M.at<uchar>(x, y)){
        case 1:
          d = abs(x - y);
          x--;
          y--;
          break;
        case 2:
          d = 0; 
          x--;
          break;
        case 3:
          y--;
          break;
      }
      dynamic_disparities.at<uchar>(y_0 - half_window_size, x) = d;
    }



  }
  std::cout << "Calculating disparities for the dynamic approach... Done.\r" << std::flush;
  std::cout << std::endl;
 
}


int Dissimilarity(
  cv::Mat& image1, cv::Mat& image2,
  int half_window_size, int y_0, int i, int j)
{
  int sum = 0;
  for (int u = -half_window_size; u <= half_window_size; ++u) {
    for (int v = -half_window_size; v <= half_window_size; ++v) {
      int i1 = image1.at<uchar>(y_0 + u, i + v);
      int i2 = image2.at<uchar>(y_0 + u, j + v);
      //sum += (i1 - i2) * (i1 - i2); //SSD
      sum += std::abs(i1 - i2); // SAD
    }
  }
  return sum;
}


void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{

  const auto& b = baseline ;
  const auto& f = focal_length ;
  std::stringstream out3d;
  out3d << output_file << ".xyz";
  std::ofstream outfile(out3d.str());
  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      if (disparities.at<uchar>(i, j) == 0) continue;

      const double d = static_cast<double>(disparities.at<uchar>(i, j) + dmin);

      const double Z = f * b / d;
      const double X = (i - width / 2) * b / d;
      const double Y = (j - height / 2) * b / d;
	  
      outfile << X << " " << Y << " " << Z << std::endl;


    }
  }
  std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
  std::cout << std::endl;
}
