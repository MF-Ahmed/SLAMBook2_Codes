#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs; // color map and depth map
    vector<Eigen::Isometry3d> poses; // camera poses

    ifstream fin("/home/user/data/git/SLAMBook2_Codes/ch12/dense_RGBD/data/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("/home/user/data/git/SLAMBook2_Codes/ch12/dense_RGBD/data/%s/%d.%s"); //image file format
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // use -1 to read the original image

        double data[7] = {0};
        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // Calculate point cloud and stitch
    // camera internal parameters
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "Converting image to point cloud..." << endl;

    // Define the format used by the point cloud: XYZRGB is used here
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // create a new point cloud
    PointCloud::Ptr pointCloud(new PointCloud);
    for (int i = 0; i < 5; i++) {
        PointCloud::Ptr current(new PointCloud);
        cout << "Converted image: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // depth value
                if (d == 0) continue; // 0 means no measurement
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                current->points.push_back(p);
            }
        // depth filter and statistical removal
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;
    cout << "point cloud total : " << pointCloud->size() << " points." << endl;

    // voxel filter
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.03;
    voxel_filter.setLeafSize(resolution, resolution, resolution); // resolution
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);

    cout << "After filtering, the point cloud has a total of : " << pointCloud->size() << " points." << endl;

    pcl::io::savePCDFileBinary("/home/user/data/git/SLAMBook2_Codes/ch12/dense_RGBD/data/map.pcd", *pointCloud);
    return 0;
}