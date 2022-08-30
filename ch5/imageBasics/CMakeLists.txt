
cmake_minimum_required(VERSION 3.0.2)
#project(imageBasics)
#use c++14 standard
set (CMAKE_CXX_FLAGS "-std=c++14")
# find OpenCV

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(imageBasics imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})

#
add_executable(undistortImage undistortImage.cpp)
target_link_libraries(undistortImage ${OpenCV_LIBS})


