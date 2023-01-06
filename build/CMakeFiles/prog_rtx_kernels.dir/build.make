# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cristobal/temporal/RTX-CUDA-template

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cristobal/temporal/RTX-CUDA-template/build

# Include any dependencies generated for this target.
include CMakeFiles/prog_rtx_kernels.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/prog_rtx_kernels.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/prog_rtx_kernels.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/prog_rtx_kernels.dir/flags.make

CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx: CMakeFiles/prog_rtx_kernels.dir/flags.make
CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx: CMakeFiles/prog_rtx_kernels.dir/includes_CUDA.rsp
CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx: /home/cristobal/temporal/RTX-CUDA-template/src/rtx_kernels.cu
CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx: CMakeFiles/prog_rtx_kernels.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cristobal/temporal/RTX-CUDA-template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx -MF CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx.d -x cu -ptx /home/cristobal/temporal/RTX-CUDA-template/src/rtx_kernels.cu -o CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx

CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

prog_rtx_kernels: CMakeFiles/prog_rtx_kernels.dir/src/rtx_kernels.ptx
prog_rtx_kernels: CMakeFiles/prog_rtx_kernels.dir/build.make
.PHONY : prog_rtx_kernels

# Rule to build all files generated by this target.
CMakeFiles/prog_rtx_kernels.dir/build: prog_rtx_kernels
.PHONY : CMakeFiles/prog_rtx_kernels.dir/build

CMakeFiles/prog_rtx_kernels.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/prog_rtx_kernels.dir/cmake_clean.cmake
.PHONY : CMakeFiles/prog_rtx_kernels.dir/clean

CMakeFiles/prog_rtx_kernels.dir/depend:
	cd /home/cristobal/temporal/RTX-CUDA-template/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cristobal/temporal/RTX-CUDA-template /home/cristobal/temporal/RTX-CUDA-template /home/cristobal/temporal/RTX-CUDA-template/build /home/cristobal/temporal/RTX-CUDA-template/build /home/cristobal/temporal/RTX-CUDA-template/build/CMakeFiles/prog_rtx_kernels.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/prog_rtx_kernels.dir/depend

