
cmake_minimum_required(VERSION 3.0.2)
#use c++14 standard
set (CMAKE_CXX_FLAGS "-std=c++14")
# find OpenCV

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("/usr/include/eigen3")
find_package(Pangolin REQUIRED)

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})