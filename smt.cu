#include <time.h>
#include <sys/time.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <openssl/modes.h>
#include "SMTLIB/Float.h"
#include "SMTLIB/BufferRef.h"
#include "cuda_aes.h"

// Threads per block
#define N 1024
// Number of blocks
#define M 65536

// Size of all variables needed by the SMT formula, in bytes
#define VARSIZE 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
 if (code != cudaSuccess) 
 {
  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  if (abort) exit(code);
 }
 else
 {
   //fprintf(stderr,"Success: %s %s %d\n", cudaGetErrorString(code), file, line);
 }
}

__device__ int solved = 0;

__device__ void LLVMFuzzerTestOneInput(const uint8_t *in_data, size_t size, uint8_t *out) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (size < VARSIZE) {
    out[i] = 0; return;
  }
  // Get our local chunk
  const uint8_t *data = in_data + (VARSIZE*i);

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
  out[i] = 1;
  solved = 1;
  return;
}

// loop:
//  run threads
//  check if any returned 1
//  mutate input buffer using AES
__global__ void fuzz(uint8_t *in_data, size_t size, const uint8_t *key, uint8_t *out) {
    for (int i = 0; i < 1000; i++) {
        LLVMFuzzerTestOneInput(in_data, size, out);
        // TODO: uhhh this is not right if VARSIZE != 16
        encrypt_k(in_data, key, N*M*16);
    }
    if (solved) return;
    return;
}

int main(int argc, char **argv) {
  uint8_t *buf;
  uint8_t *obuf;
  uint8_t *gbuf;
  uint8_t *gobuf;

  // Initialize the input array. We use AES-128 in CTR mode because it's much
  // faster than reading from /dev/urandom.
  AES_KEY key;
  unsigned char ckey[16];
  unsigned char iv[16];
  RAND_bytes(ckey, 16);
  RAND_bytes(iv, 8);
  memset(iv+8, 0, 8);
  unsigned char ecount[AES_BLOCK_SIZE] = {};
  unsigned int num = 0;
  AES_set_encrypt_key(ckey, 128, &key);

  // Pre-expand the round keys and copy to device mem
  uint8_t rkey[176];
  const uint8_t *drkey;
  expand_key(ckey, rkey);
  gpuErrchk(cudaMalloc(&drkey, sizeof(uint8_t) * 176));
  gpuErrchk(cudaMemcpy((uint8_t *)drkey, rkey, sizeof(uint8_t) * 176, cudaMemcpyHostToDevice));

  // Generate initial random buf on host side using AES-CTR from OpenSSL
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  printf("Generating %d random bytes...\n", 16*N*M);
  buf = (uint8_t *)malloc(VARSIZE*N*M);
  // TODO figure out padding if VARSIZE is not 16
  for (int i = 0; i < VARSIZE*N*M; i += 16) {
    CRYPTO_ctr128_encrypt(buf+i, buf+i, AES_BLOCK_SIZE, &key, iv, ecount, &num, (block128_f)AES_encrypt);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
  printf("Initialized random data in %lu microseconds.\n", delta_us);

  // Alloc GPU buffers
  printf("Alloc GPU buffers and copy...\n");
  printf("Attempting to allocate %zu bytes on the GPU\n", VARSIZE*N*M*sizeof(uint8_t));
  gpuErrchk(cudaMalloc(&gbuf, VARSIZE*N*M*sizeof(uint8_t)));
  printf("Attempting to allocate %zu bytes on the GPU\n", N*M*sizeof(uint8_t));
  gpuErrchk(cudaMalloc(&gobuf, N*M*sizeof(uint8_t)));
  printf("Copying %zu bytes from host buffer to GPU buffer\n", VARSIZE*N*M*sizeof(uint8_t));
  gpuErrchk(cudaMemcpy(gbuf, buf, VARSIZE*N*M*sizeof(uint8_t), cudaMemcpyHostToDevice));

  // Start fuzzing!
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  printf("Launching kernel...\n");
  fuzz<<<M,N>>>(gbuf, VARSIZE, drkey, gobuf);
  // Get and print output
  obuf = (uint8_t *)malloc(N*M*sizeof(uint8_t));
  gpuErrchk(cudaMemcpy(obuf, gobuf, N*M*sizeof(uint8_t), cudaMemcpyDeviceToHost));
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
  printf("Ran %lu executions in %lu microseconds (%f execs/s).\n", N*M*1000UL, delta_us, (double)(N*M*1000UL)/(delta_us / 1000000.0));
  for (int i = 0; i < N*M; i++) {
	if (obuf[i]) {
      printf("Found a satisfying assignment:\n");
      for (int j = 0; j < VARSIZE; j++) printf("%02x", buf[VARSIZE*i+j]); printf("\n");
      break;
    }
  }
}
