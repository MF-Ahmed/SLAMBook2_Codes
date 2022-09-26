set(CMAKE_BUILD_TYPE "Release")
# Add c++11 standard support
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

#Eigen
include_directories("/usr/include/eigen3")

# Find OpenCV library
find_package(OpenCV REQUIRED)
# add header file
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)