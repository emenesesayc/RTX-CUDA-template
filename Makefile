OPTIX_HOME = /opt/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64
FLAGS = -arch=sm_75 -m64 --use_fast_math

all:
	mkdir -p ptx
	nvcc $(FLAGS) --ptx -I$(OPTIX_HOME)/include src/rtx_kernels.cu -o ptx/rtx_kernels.ptx
	nvcc $(FLAGS) -rdc=true -I$(OPTIX_HOME)/include -I. common/common.cpp src/main.cu -o rtxcuda
