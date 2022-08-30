#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

string image_path {"/home/uzi/Data/AllGit/SLAMBook2_Codes/ch5/imageBasics/ubuntu.png"};
int main(int argc, char **argv) {
  // Read the image specified by argv[1]
  cv::Mat image;
  image = cv::imread(image_path); //cv::imread function reads the image under the specified path

// Determine if the image file is read correctly
  if (image.data == nullptr) { //The data does not exist, maybe the file does not exist
    cerr << "file" << image_path << "does not exist." << endl;
    return 0;
  }

  // The file is read smoothly, first output some basic information
  cout << "The width of the image is " << image.cols << ", the height is " << image.rows << ", the number of channels is " << image.channels() << endl;
  cv::imshow("image", image); // show the image with cv::imshow
  cv::waitKey(0); // Pause the program and wait for a key input

  // Determine the type of image
  if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
    // The image type does not meet the requirements
    cout << "Please enter a color or grayscale image." << endl;
    return 0;
  }

  // Traverse the image, please note that the following traversal methods can also be used for random pixel access
  // use std::chrono to time the algorithm
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (size_t y = 0; y < image.rows; y++) {
    // Get the row pointer of the image with cv::Mat::ptr
    unsigned char *row_ptr = image.ptr<unsigned char>(y); // row_ptr is the head pointer of row y
    for (size_t x = 0; x < image.cols; x++) {
      // access the pixel at x,y
      unsigned char *data_ptr = &row_ptr[x * image.channels()]; // data_ptr points to the pixel data to be accessed
      // Output each channel of the pixel, if it is a grayscale image, there is only one channel
      for (int c = 0; c != image.channels(); c++) {
        unsigned char data = data_ptr[c]; // data is the value of the cth channel of I(x,y)
      }
    }
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
  cout << "Time to traverse the image:" << time_used.count() << "Seconds." << endl;

  // Copy about cv::Mat
  // Direct assignment does not copy the data
  cv::Mat image_another = image;
  // Modifying image_another will cause the image to change
  image_another(cv::Rect(0, 0, 100, 100)).setTo(0); // zero out the 100*100 block in the upper left corner
  cv::imshow("image", image);
  cv::waitKey(0);

  // use the clone function to copy the data
  cv::Mat image_clone = image.clone();
  image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
  cv::imshow("image", image);
  cv::imshow("image_clone", image_clone);
  cv::waitKey(0);

  // There are many basic operations for images, such as cutting, rotating, scaling, etc., 
  //which will not be introduced one by one due to space limitations. Please refer to the 
  //official OpenCV documentation to query the calling method of each function.
  cv::destroyAllWindows();
  return 0;
}