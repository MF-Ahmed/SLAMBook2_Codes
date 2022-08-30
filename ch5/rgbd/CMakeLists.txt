
cmake_minimum_required(VERSION 3.0.2)
#project(imageBasics)
#use c++14 standard
set (CMAKE_CXX_FLAGS "-std=c++17")
# find OpenCV

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("/usr/include/eigen3")

# Eigen
include_directories("/usr/include/eigen3")
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})


