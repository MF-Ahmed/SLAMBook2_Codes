//
// Created by gaoxiang on 19-5-2.
//
#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {

struct Frame;
struct MapPoint;

/**
  * 2D feature points
  * Will be associated with a map point after triangulation
  */
struct Feature {
    public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
     typedef std::shared_ptr<Feature> Ptr;

     std::weak_ptr<Frame> frame_; // The frame that holds the feature
     cv::KeyPoint position_; // 2D extraction position
     std::weak_ptr<MapPoint> map_point_; // Associate map point

     bool is_outlier_ = false; // Whether it is an outlier
     bool is_on_left_image_ = true; // Whether the logo is mentioned on the left image, false is the right image
   public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
};
}  // namespace myslam

#endif  // MYSLAM_FEATURE_H
