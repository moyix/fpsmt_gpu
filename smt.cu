#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#if RNG == CURAND
// It won't build unless this include is on this line. I have no idea why.
#include "curand_kernel.h"
#endif
#include "SMTLIB/BufferRef.h"
#include "SMTLIB/Float.h"
#include "smt.h"
#include "theory.h"

#if RNG == AES

#include "cuda_aes.h"
#define AES_BLOCK_SIZE 16

__host__ __device__ inline int64_t aes_pad(int64_t num) { return (num + AES_BLOCK_SIZE - 1) & -AES_BLOCK_SIZE; }

#elif RNG == CHAM

#include "cham.h"
#include "cuda_aes.h"
#define AES_BLOCK_SIZE 16

__host__ __device__ inline int64_t aes_pad(int64_t num) { return (num + AES_BLOCK_SIZE - 1) & -AES_BLOCK_SIZE; }

#endif

#ifndef ITERS
#define ITERS 1000
#endif

#define RESULTS_FNAME "results.csv"

// should come from theory.cu
extern int varsize;

__device__ int solved = 0;
int host_solved = 0;
volatile int finished_dev = 0;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  } else {
    // fprintf(stderr,"Success: %s %s %d\n", cudaGetErrorString(code), file,
    // line);
  }
}

// Note: size is the *unpadded* size of the input vars
//__global__ void fuzz(uint8_t *in_data, size_t size, const uint8_t *key, uint64_t *gobuf, unsigned long long *execs) {
#if RNG == CURAND
__global__ void fuzz(uint8_t *in_data, size_t size, curandState *state, uint64_t *gobuf, unsigned long long *execs) {
#elif RNG == AES
__global__ void fuzz(uint8_t *in_data, size_t size, const uint8_t *key, uint64_t *gobuf, unsigned long long *execs) {
#elif RNG == CHAM
__global__ void fuzz(uint8_t *in_data, size_t size, const uint8_t *key, uint64_t *gobuf, unsigned long long *execs) {
#endif
  extern __shared__ uint8_t sdata[];

  int bindex = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t soff;
  uint64_t offset;

#if RNG == CURAND
  soff = threadIdx.x * size;
  offset = bindex * size;

  uint8_t *data = sdata + soff;
  uint64_t seed = bindex * 37;
  curand_init(seed, bindex, 0, &state[bindex]);
  curandState localState = state[bindex];
  if (LLVMFuzzerTestOneInput(data, size)) {
    *gobuf = bindex;
    memcpy(in_data + offset, sdata + soff, size);
    solved = 1;
  }
#elif RNG == AES
  int64_t padded = aes_pad(size);
  offset = bindex * padded;

  // First time initialize block to i
  for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
    *(uint64_t *)(sdata + soff + i) = bindex * (padded / AES_BLOCK_SIZE) + i;
  }
  if (LLVMFuzzerTestOneInput(sdata + soff, size)) {
    *gobuf = bindex;
    memcpy(in_data + offset, sdata + soff, size);
    solved = 1;
  }
  // Add increment to randomize (I hope?)
  for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
    *(uint64_t *)(sdata + soff + i) = bindex * (padded / AES_BLOCK_SIZE) + i;
  }
#elif RNG == CHAM
  offset = bindex * size;
  uint8_t rks[4 * 16] = {0};
  int64_t padded = aes_pad(size);
  cham128_keygen(rks, key);

  // Get our local chunk
  soff = threadIdx.x * padded;

  // First time initialize block to i
  for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
    *(uint64_t *)(sdata + soff + i) = bindex * (padded / AES_BLOCK_SIZE) + i;
  }
#endif

  atomicAdd(execs, ITERS);
  for (int i = 0; i < ITERS; i++) {
    // Randomize input for our slice

#if RNG == CURAND
    // TODO: once we confirm 16bytes and we generate 8bytes, replace loop with writes
    uint8_t *curr = data;
    while (curr < data + size) {
      *curr++ = curand(&localState); // TODO: i think this is 8bytes but not sure, alternative is uint4
    }
#elif RNG == AES
    for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
      encrypt_one_table(sdata + soff, key, i);
    }
#elif RNG == CHAM
    for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
      cham128_encrypt(sdata + soff, sdata + soff, rks);
    }
    if (LLVMFuzzerTestOneInput(sdata + soff, size)) {
      *gobuf = bindex;
      memcpy(in_data + offset, sdata + soff, size);
      solved = 1;
    }
    // Add increment to randomize (I hope?)
    for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
      *(uint64_t *)(sdata + soff + i) = bindex * (padded / AES_BLOCK_SIZE) + i;
    }
#endif
  }
}

void CUDART_CB finishedCB(void *data) { finished_dev = *(int *)data; }

void launch_kernel(int device, int varsize, uint8_t **ret_gbuf, uint64_t **ret_gobuf, unsigned long long **ret_execs) {
  cudaSetDevice(device);

  uint8_t *gbuf;
  uint64_t *gobuf;
  unsigned long long *gexecs;

  int size = varsize; // i think?
#if RNG == CURAND
  curandState *rngStates;
  gpuErrchk(cudaMalloc(&rngStates, N * M * sizeof(curandState)));

#elif RNG == AES
  int64_t padded = aes_pad(varsize);
  printf("Padding varsize from %d to %ld\n", varsize, padded);
  unsigned char ckey[AES_BLOCK_SIZE];
  FILE *rng = fopen("/dev/urandom", "rb");
  fread(ckey, AES_BLOCK_SIZE, 1, rng);
  fclose(rng);

  // Pre-expand the round keys and copy to device mem
  uint8_t rkey[176];
  const uint8_t *drkey;
  expand_key(ckey, rkey);
  gpuErrchk(cudaMalloc(&drkey, 176));
  gpuErrchk(cudaMemcpy((uint8_t *)drkey, rkey, sizeof(uint8_t) * 176, cudaMemcpyHostToDevice));

#elif RNG == CHAM
  int64_t padded = aes_pad(varsize);
  printf("Padding varsize from %d to %ld\n", varsize, padded);
  unsigned char ckey[AES_BLOCK_SIZE];
  FILE *rng = fopen("/dev/urandom", "rb");
  fread(ckey, AES_BLOCK_SIZE, 1, rng);
  fclose(rng);

  const uint8_t *dkey;
  gpuErrchk(cudaMalloc(&dkey, 16));
  gpuErrchk(cudaMemcpy((uint8_t *)dkey, ckey, 16, cudaMemcpyHostToDevice));
#endif

  // Alloc GPU buffers
  gpuErrchk(cudaMalloc(&gbuf, size * N * M));
  gpuErrchk(cudaMalloc(&gobuf, sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&gexecs, sizeof(unsigned long long)));

  *ret_gbuf = gbuf;
  *ret_gobuf = gobuf;
  *ret_execs = gexecs;

  // Start fuzzing!
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreate(&stream));
  int *dev = (int *)malloc(sizeof(int));
  *dev = device + 1;
  printf("Launching kernel on GPU%d...\n", device);
#if RNG == CURAND
  fuzz<<<M, N, N * size, stream>>>(gbuf, varsize, rngStates, gobuf, gexecs);
#elif RNG == AES
  fuzz<<<M, N, N * padded, stream>>>(gbuf, varsize, drkey, gobuf, gexecs);
#elif RNG == CHAM
  fuzz<<<M, N, N * padded, stream>>>(gbuf, varsize, dkey, gobuf, gexecs);
#endif
  cudaMemcpy(&host_solved, &solved, sizeof(int), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaLaunchHostFunc(stream, finishedCB, dev));
}

int main(int argc, char **argv) {
  int NUM_GPU;
  gpuErrchk(cudaGetDeviceCount(&NUM_GPU));
  if (NUM_GPU < 1) {
    fprintf(stderr, "No CUDA-capable GPUs detected!\n");
    return 1;
  }

  printf("Running %d iters\n", ITERS);

  uint8_t *gbuf[NUM_GPU];
  uint64_t *gobuf[NUM_GPU];
  unsigned long long *goexecs[NUM_GPU];

  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
  for (int i = 0; i < NUM_GPU; i++) {
    launch_kernel(i, varsize, &gbuf[i], &gobuf[i], &goexecs[i]);
  }

  printf("Waiting on GPUs...\n");
  while (!finished_dev)
    sched_yield();
  int i = finished_dev - 1;
  // Wait to finish
  cudaSetDevice(i);
  cudaDeviceSynchronize();
  unsigned long long hexecs;
  printf("Search completed on device %d\n", i);
  gpuErrchk(cudaMemcpy(&hexecs, goexecs[i], sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  float seconds = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + (end.tv_sec - begin.tv_sec);
  printf("Did %llu execs in %f seconds, %f execs/s\n", hexecs, seconds, hexecs / seconds);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  FILE *results_fd;
  if (access(RESULTS_FNAME, F_OK) == 0) {
    results_fd = fopen(RESULTS_FNAME, "a");
  } else {
    results_fd = fopen(RESULTS_FNAME, "w");
    fprintf(results_fd, "RNG,execs,seconds,execsps,iters,threads per block,number of blocks,dev name,dev mem rate (KHz),dev bus width (bits),dev peak mem bandwidth (GB/s)\n"); // write headers
  }

#if RNG == CURAND
  const char* rngname = "CURAND";
#elif RNG == AES
  const char* rngname = "AES";
#elif RNG == CHAM
  const char* rngname = "CHAM";
#endif

  printf("writing results to ");
  printf(RESULTS_FNAME);
  printf("\n");
  fprintf(results_fd, "%s,%llu,%f,%f,%d,%d,%s,%d,%d,%f\n", rngname, hexecs, seconds, hexecs/seconds,
          N, M,
          prop.name, prop.memoryClockRate, prop.memoryBusWidth,
          2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  fclose(results_fd);

#if RNG == CURAND
  if (host_solved) {
    // Get and print output
    uint8_t *buf = (uint8_t *)malloc(varsize);
    uint64_t oindex;
    gpuErrchk(cudaMemcpy(&oindex, gobuf[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(buf, gbuf[i] + (oindex * varsize), varsize, cudaMemcpyDeviceToHost));
    printf("Found a satisfying assignment on device %d thread %lu:\n", i, oindex);
    for (int k = 0; k < varsize; k++)
      printf("%02x", buf[k]);
    printf("\n");
  } else {
    fprintf(stderr, "No satisfying assignment found");
  }
#endif
}
