# declare the required minimum version of cmake
cmake_minimum_required(VERSION 2.8)

# declare a cmake project
project(HelloSLAM)

# Set the compilation mode
set(CMAKE_BUILD_TYPE "Debug")

# add an executable
# Syntax: add_executable( program name source code file )
add_executable(helloSLAM helloSLAM.cpp)

# Add hello library
add_library(hello libHelloSLAM.cpp)
# shared library
add_library(hello_shared SHARED libHelloSLAM.cpp)

# Add an executable program to call the function in the hello library
add_executable(useHello useHello.cpp)
# Link the library file to the executable program
target_link_libraries(useHello hello_shared)
