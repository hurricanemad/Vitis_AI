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
CMAKE_SOURCE_DIR = /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/build

# Include any dependencies generated for this target.
include CMakeFiles/OFClassificationPreProcess.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/OFClassificationPreProcess.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/OFClassificationPreProcess.dir/flags.make

CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.o: CMakeFiles/OFClassificationPreProcess.dir/flags.make
CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.o: ../AksOFClassificationPreProcess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.o -c /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/AksOFClassificationPreProcess.cpp

CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/AksOFClassificationPreProcess.cpp > CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.i

CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/AksOFClassificationPreProcess.cpp -o CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.s

# Object files for target OFClassificationPreProcess
OFClassificationPreProcess_OBJECTS = \
"CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.o"

# External object files for target OFClassificationPreProcess
OFClassificationPreProcess_EXTERNAL_OBJECTS =

libOFClassificationPreProcess.so.1.3.0: CMakeFiles/OFClassificationPreProcess.dir/AksOFClassificationPreProcess.cpp.o
libOFClassificationPreProcess.so.1.3.0: CMakeFiles/OFClassificationPreProcess.dir/build.make
libOFClassificationPreProcess.so.1.3.0: CMakeFiles/OFClassificationPreProcess.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libOFClassificationPreProcess.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OFClassificationPreProcess.dir/link.txt --verbose=$(VERBOSE)
	$(CMAKE_COMMAND) -E cmake_symlink_library libOFClassificationPreProcess.so.1.3.0 libOFClassificationPreProcess.so.1 libOFClassificationPreProcess.so

libOFClassificationPreProcess.so.1: libOFClassificationPreProcess.so.1.3.0
	@$(CMAKE_COMMAND) -E touch_nocreate libOFClassificationPreProcess.so.1

libOFClassificationPreProcess.so: libOFClassificationPreProcess.so.1.3.0
	@$(CMAKE_COMMAND) -E touch_nocreate libOFClassificationPreProcess.so

# Rule to build all files generated by this target.
CMakeFiles/OFClassificationPreProcess.dir/build: libOFClassificationPreProcess.so

.PHONY : CMakeFiles/OFClassificationPreProcess.dir/build

CMakeFiles/OFClassificationPreProcess.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/OFClassificationPreProcess.dir/cmake_clean.cmake
.PHONY : CMakeFiles/OFClassificationPreProcess.dir/clean

CMakeFiles/OFClassificationPreProcess.dir/depend:
	cd /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/build /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/build /workspace/examples/Whole-App-Acceleration/apps/fall_detection/kernel_src/of_classification_preprocess/build/CMakeFiles/OFClassificationPreProcess.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/OFClassificationPreProcess.dir/depend

