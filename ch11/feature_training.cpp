#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/**************************************************** **
  * This section demonstrates how to train a dictionary from ten images in the data/ directory
  *****************************************************/

int main( int argc, char** argv ) {
    // read the image 
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    string abs_path = "/home/user/data/git/SLAMBook2_Codes/ch11";
    for ( int i=0; i<10; i++ )
    {
        string path =  abs_path+"/data/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }
    
    // create vocabulary 
    //cout<<"\n\r creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    //cout<<"\n\r vocabulary info: "<<vocab<<endl;
    vocab.save( abs_path+"/myvocabulary.yml.gz" );
    cout<<"done!"<<endl;
    
    return 0;
}