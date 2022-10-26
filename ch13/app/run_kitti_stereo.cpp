//
// Created by gaoxiang on 19-5-4.
//
// Modified by Farhan on 26-10-2022

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

DEFINE_string(config_file, "/home/user/data/git/SLAMBook2_Codes/ch13/config/default.yaml", "config file path");

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();
    return 0;
}
