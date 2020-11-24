SMTLIB/%.o: SMTLIB/%.cu
	$(MAKE) -C SMTLIB

INCLUDE = -I.
NVCC_FLAGS = --expt-relaxed-constexpr -DJFS_RUNTIME_FAILURE_CALLS_ABORT -dc -std=c++11 $(INCLUDE)

%.o: %.cu
	nvcc $(NVCC_FLAGS) -c $< -o $@

smtlib-objs =  SMTLIB/Core.o
smtlib-objs += SMTLIB/Logger.o
smtlib-objs += SMTLIB/NativeFloat.o
smtlib-objs += SMTLIB/Messages.o
smtlib-objs += SMTLIB/Float.o
smtlib-objs += SMTLIB/NativeBitVector.o

smt: smt.o aes.o $(smtlib-objs)
	nvcc $^ -o $@ -lcrypto

all: smt

clean:
	rm -f *.o SMTLIB/*.o smt
