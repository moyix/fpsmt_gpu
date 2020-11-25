#ifndef THEORY_H
#define THEORY_H

#include "SMTLIB/BufferRef.h"
#include "SMTLIB/Float.h"
#include <stdio.h>

extern __device__ int solved;
extern volatile int finished_dev;

// Size of all variables needed by the SMT formula, in bytes
__device__ int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

#endif
