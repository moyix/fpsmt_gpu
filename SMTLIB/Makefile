INCLUDE = -I..
NVCC_FLAGS = --expt-relaxed-constexpr -DJFS_RUNTIME_FAILURE_CALLS_ABORT -dc -std=c++11 $(INCLUDE)

all: Core.o Float.o Logger.o Messages.o NativeBitVector.o NativeFloat.o

%.o : %.cu
	nvcc $(NVCC_FLAGS) -c $< -o $@
