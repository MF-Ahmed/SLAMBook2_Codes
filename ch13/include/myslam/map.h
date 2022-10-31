#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam {
/**
 * @brief map
 * Interaction with the map: the front-end calls InsertKeyframe and InsertMapPoint to insert new frames and map points, the back-end maintains the structure of the map, determines outlier/elimination, etc.
 */
class Map {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    Map() {}

    /// Add a keyframe
    void InsertKeyFrame(Frame::Ptr frame);
    /// Add a map vertex
    void InsertMapPoint(MapPoint::Ptr map_point);

    /// Get all map points
    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    /// Get all keyframes
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    /// Get the active map point
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    /// Get the activation keyframe
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    /// Clean up the points in the map where the number of observations is zero
    void CleanMap();

   private:
    // Make old keyframes inactive
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    LandmarksType landmarks_; // all landmarks
    LandmarksType active_landmarks_; // active landmarks
    KeyframesType keyframes_; // all key-frames
    KeyframesType active_keyframes_; // all key-frames

    Frame::Ptr current_frame_ = nullptr;

    // settings
    int num_active_keyframes_ = 10; // number of active keyframes changed from 7 to 10
};
}  // namespace myslam

#endif  // MAP_H
