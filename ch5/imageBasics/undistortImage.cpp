#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "/home/uzi/Data/AllGit/SLAMBook2_Codes/ch5/imageBasics/distorted.png"; // make sure the path is correct

int main(int argc, char **argv) {

  // This program implements the code for the de-distortion part. Although we can call OpenCV's dewarping, it's helpful to implement it ourselves.
  // Distortion parameters
  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
  // internal parameter
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

  cv::Mat image = cv::imread(image_file, 0); // image is grayscale, CV_8UC1
  int rows = image.rows, cols = image.cols;
  cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1); // Image after undistortion

  // Calculate the content of the dedistorted image
  for (int v = 0; v < rows; v++) {
    for (int u = 0; u < cols; u++) {
      // According to the formula, the calculated point (u, v) corresponds to the coordinates in the distorted image (u_distorted, v_distorted)
      double x = (u - cx) / fx, y = (v - cy) / fy;
      double r = sqrt(x * x + y * y);
      double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
      double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
      double u_distorted = fx * x_distorted + cx;
      double v_distorted = fy * y_distorted + cy;

      // assignment (nearest neighbor interpolation)
      if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
        image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
      } else {
        image_undistort.at<uchar>(v, u) = 0;
      }
    }
  }

  // draw the image after distortion //

  cv::imshow("distorted", image);
  cv::imshow("undistorted", image_undistort);
  cv::waitKey();
  return 0;
}