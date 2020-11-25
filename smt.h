#ifndef SMT_H
#define SMT_H

#include "SMTLIB/BufferRef.h"
#include "SMTLIB/Float.h"
#include "cuda_aes.h"

// Threads per block
#define N 1024
// Number of blocks
#define M 65536

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#endif
