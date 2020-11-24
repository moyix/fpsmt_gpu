#include <time.h>
#include <sys/time.h>
#include <sched.h>
#include "SMTLIB/Float.h"
#include "SMTLIB/BufferRef.h"
#include "cuda_aes.h"

#define NUM_GPU 2

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
    while (!solved) {
        // TODO: uhhh this is not right if VARSIZE != 16
        encrypt_k(in_data, key, N*M*16);
        LLVMFuzzerTestOneInput(in_data, size, out);
    }
    return;
}

volatile int finished_dev = 0;

void CUDART_CB finishedCB(void *data) {
  finished_dev = *(int *)data;
}

void launch_kernel(int device, uint8_t **ret_gbuf, uint8_t **ret_gobuf) {
  cudaSetDevice(device);

  uint8_t *buf;
  uint8_t *gbuf;
  uint8_t *gobuf;

  //AES_KEY key;
  unsigned char ckey[16];
  FILE *rng = fopen("/dev/urandom","rb");
  fread(ckey, 16, 1, rng);
  fclose(rng);

  // Initialize the input array. We use AES-128 in CTR mode because it's much
  // faster than reading from /dev/urandom.
  // Pre-expand the round keys and copy to device mem
  uint8_t rkey[176];
  const uint8_t *drkey;
  expand_key(ckey, rkey);
  gpuErrchk(cudaMalloc(&drkey, sizeof(uint8_t) * 176));
  gpuErrchk(cudaMemcpy((uint8_t *)drkey, rkey, sizeof(uint8_t) * 176, cudaMemcpyHostToDevice));

  // Generate initial random buf on host side using AES-CTR from OpenSSL
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  buf = (uint8_t *)malloc(VARSIZE*N*M);
  // TODO figure out padding if VARSIZE is not 16
  for (uint64_t i = 0; i < VARSIZE*N*M; i += 16) {
    *(uint64_t *)(buf+i) = i;
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

  *ret_gbuf = gbuf;
  *ret_gobuf = gobuf;

  // Start fuzzing!
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreate(&stream));
  int *dev = (int *)malloc(sizeof(int));
  *dev = device + 1;
  printf("Launching kernel on GPU%d...\n", device);
  fuzz<<<M,N,0,stream>>>(gbuf, VARSIZE, drkey, gobuf);
  gpuErrchk(cudaLaunchHostFunc(stream, finishedCB, dev));

  return;
}

int main(int argc, char **argv) {
  uint8_t *buf[NUM_GPU];
  uint8_t *obuf[NUM_GPU];
  uint8_t *gbuf[NUM_GPU];
  uint8_t *gobuf[NUM_GPU];
  
  for (int i = 0; i < NUM_GPU; i++) {
    launch_kernel(i, &gbuf[i], &gobuf[i]);
  }

  while (!finished_dev) sched_yield();
  int i = finished_dev - 1;
  printf("Search completed on device %d\n", i);

  // Get and print output
  buf[i] = (uint8_t *)malloc(VARSIZE*N*M*sizeof(uint8_t));
  obuf[i] = (uint8_t *)malloc(N*M*sizeof(uint8_t));
  gpuErrchk(cudaMemcpy(buf[i], gbuf[i], VARSIZE*N*M*sizeof(uint8_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(obuf[i], gobuf[i], N*M*sizeof(uint8_t), cudaMemcpyDeviceToHost));
  for (int j = 0; j < N*M; j++) {
    if (obuf[i][j]) {
      printf("Found a satisfying assignment on device %d thread %d:\n", i, j);
      for (int k = 0; k < VARSIZE; k++) printf("%02x", buf[i][VARSIZE*j+k]); printf("\n");
      break;
    }
  }
}
