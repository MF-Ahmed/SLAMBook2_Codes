cmake_minimum_required( VERSION 2.8 )
project( visualizeGeometry )

set(CMAKE_CXX_FLAGS "-std=c++14")

# can also set to           set(CMAKE_CXX_FLAGS "-std=c++17")
# but can not set to           set(CMAKE_CXX_FLAGS "-std=c++11")  on  ubunto 20.4 

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

# 添加Pangolin依赖
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )


add_executable( visualizeGeometry visualizeGeometry.cpp )
target_link_libraries( visualizeGeometry ${Pangolin_LIBRARIES} )
