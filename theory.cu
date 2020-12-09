#include "theory.h"
// #include "SMTLIB/Core.h"
// #include "SMTLIB/BufferRef.h"
// #include "SMTLIB/Float.h"
// #include "SMTLIB/Messages.h"
// #include <stdint.h>
// #include <stdlib.h>
__device__ int LLVMFuzzerTestOneInput(float x, size_t size) {
  uint64_t jfs_num_const_sat = 0;
  unsigned int sign = 0;
  unsigned int exponent = 127;
  unsigned int significand = 0;
  uint32_t rawBits = significand;
  rawBits |= (exponent << 23);
  rawBits |= (sign << 31);
  float y = *(float *)&rawBits;
  const bool jfs_ssa_4 = (x == y);
  if (jfs_ssa_4) {
    ++jfs_num_const_sat;
  }
  if (jfs_num_const_sat == 1) {
    // Fuzzing target
    return 1;
  } else {
    return 0;
  }
}
// End program
int varsize = 4;
