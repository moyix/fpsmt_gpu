#include "SMTLIB/Float.h"
#include "SMTLIB/BufferRef.h"

__global__ void LLVMFuzzerTestOneInput(const uint8_t *data, size_t size, uint8_t *out) {
  int i = threadIdx.x;
  if (size < 16) {
    out[i] = 0; return;
  }
  // Get our local chunk
  data = data + (16*i);

  BufferRef<const uint8_t> jfs_buffer_ref =
      BufferRef<const uint8_t>(data, size);
  const Float<11, 53> a = makeFloatFrom<11, 53>(jfs_buffer_ref, 0, 63);
  const Float<11, 53> b = makeFloatFrom<11, 53>(jfs_buffer_ref, 64, 127);
  const bool jfs_ssa_0 = a.isNaN();
  const bool jfs_ssa_1 = !(jfs_ssa_0);
  if (jfs_ssa_1) {
  } else {
    out[i] = 0; return;
  }
  const bool jfs_ssa_2 = b.isNaN();
  const bool jfs_ssa_3 = !(jfs_ssa_2);
  if (jfs_ssa_3) {
  } else {
    out[i] = 0; return;
  }
  const Float<11, 53> jfs_ssa_4 = a.div(JFS_RM_RNE, b);
  const Float<11, 53> jfs_ssa_5 = a.div(JFS_RM_RTP, b);
  const bool jfs_ssa_6 = jfs_ssa_4.ieeeEquals(jfs_ssa_5);
  const bool jfs_ssa_7 = !(jfs_ssa_6);
  if (jfs_ssa_7) {
  } else {
    out[i] = 0; return;
  }
  const bool jfs_ssa_8 = jfs_ssa_4.isNaN();
  const bool jfs_ssa_9 = !(jfs_ssa_8);
  if (jfs_ssa_9) {
  } else {
    out[i] = 0; return;
  }
  const bool jfs_ssa_10 = jfs_ssa_5.isNaN();
  const bool jfs_ssa_11 = !(jfs_ssa_10);
  if (jfs_ssa_11) {
  } else {
    out[i] = 0; return;
  }
  // Fuzzing target
  out[i] = 1; return;
}

#define N 32 
int main(int argc, char **argv) {
  uint8_t *buf;
  uint8_t obuf[N] = {};
  uint8_t *gbuf;
  uint8_t *gobuf;

  buf = (uint8_t *)malloc(16*N);
  FILE *f = fopen("/dev/urandom", "rb");
  fread(buf, 16, N, f);
  fclose(f);

  // Alloc GPU buffers
  cudaMalloc(&gbuf, 16*N);
  cudaMalloc(&gobuf, N*sizeof(uint8_t));
  cudaMemcpy(gbuf, buf, 16*N, cudaMemcpyHostToDevice);

  LLVMFuzzerTestOneInput<<<1,32>>>(gbuf, 16, gobuf);

  // Get and print output
  cudaMemcpy(obuf, gobuf, N, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) printf("%d ", obuf[i]);
  printf("\n");
}
