# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/user/All_Data/All_git/SLAMBook2_Codes/ch7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/build

# Include any dependencies generated for this target.
include CMakeFiles/pose_estimation_3d3d.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pose_estimation_3d3d.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pose_estimation_3d3d.dir/flags.make

CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.o: CMakeFiles/pose_estimation_3d3d.dir/flags.make
CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.o: ../pose_estimation_3d3d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/user/All_Data/All_git/SLAMBook2_Codes/ch7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.o -c /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/pose_estimation_3d3d.cpp

CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/pose_estimation_3d3d.cpp > CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.i

CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/pose_estimation_3d3d.cpp -o CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.s

# Object files for target pose_estimation_3d3d
pose_estimation_3d3d_OBJECTS = \
"CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.o"

# External object files for target pose_estimation_3d3d
pose_estimation_3d3d_EXTERNAL_OBJECTS =

pose_estimation_3d3d: CMakeFiles/pose_estimation_3d3d.dir/pose_estimation_3d3d.cpp.o
pose_estimation_3d3d: CMakeFiles/pose_estimation_3d3d.dir/build.make
pose_estimation_3d3d: /usr/local/lib/libopencv_gapi.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_stitching.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_alphamat.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_aruco.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_barcode.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_bgsegm.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_bioinspired.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_ccalib.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_dnn_objdetect.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_dnn_superres.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_dpm.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_face.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_freetype.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_fuzzy.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_hdf.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_hfs.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_img_hash.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_intensity_transform.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_line_descriptor.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_mcc.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_quality.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_rapid.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_reg.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_rgbd.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_saliency.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_stereo.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_structured_light.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_superres.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_surface_matching.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_tracking.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_videostab.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_viz.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_wechat_qrcode.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_xfeatures2d.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_xobjdetect.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_xphoto.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_shape.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_highgui.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_datasets.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_plot.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_text.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_ml.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_phase_unwrapping.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_optflow.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_ximgproc.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_video.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_videoio.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_objdetect.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_calib3d.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_dnn.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_features2d.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_flann.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_photo.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_imgproc.so.4.6.0
pose_estimation_3d3d: /usr/local/lib/libopencv_core.so.4.6.0
pose_estimation_3d3d: CMakeFiles/pose_estimation_3d3d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/user/All_Data/All_git/SLAMBook2_Codes/ch7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pose_estimation_3d3d"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pose_estimation_3d3d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pose_estimation_3d3d.dir/build: pose_estimation_3d3d

.PHONY : CMakeFiles/pose_estimation_3d3d.dir/build

CMakeFiles/pose_estimation_3d3d.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pose_estimation_3d3d.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pose_estimation_3d3d.dir/clean

CMakeFiles/pose_estimation_3d3d.dir/depend:
	cd /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/All_Data/All_git/SLAMBook2_Codes/ch7 /home/user/All_Data/All_git/SLAMBook2_Codes/ch7 /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/build /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/build /home/user/All_Data/All_git/SLAMBook2_Codes/ch7/build/CMakeFiles/pose_estimation_3d3d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pose_estimation_3d3d.dir/depend

