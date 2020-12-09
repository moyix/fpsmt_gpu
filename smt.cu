#include <time.h>
#include <sys/time.h>
#include <sched.h>
#include "SMTLIB/Float.h"
#include "SMTLIB/BufferRef.h"
#include "theory.h"
#include "smt.h"
#include "rot.h"

#define AES_BLOCK_SIZE 16

// should come from theory.cu
extern int varsize;

__device__ int solved = 0;
volatile int finished_dev = 0;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  } else {
    // fprintf(stderr,"Success: %s %s %d\n", cudaGetErrorString(code), file,
    // line);
  }
}

__host__ __device__ inline int64_t aes_pad(int64_t num) {
  return (num + AES_BLOCK_SIZE - 1) & -AES_BLOCK_SIZE;
}

__device__ void cham128_keygen(uint8_t* rks, const uint8_t* mk)
{
    const uint32_t* key = (uint32_t*) mk;
    uint32_t* rk = (uint32_t*) rks;

    for (size_t i = 0; i < 4; ++i) {
        rk[i] = key[i] ^ rol32(key[i], 1);
        rk[(i+4)^(0x1)] = rk[i] ^ rol32(key[i], 11);
        rk[i] ^= rol32(key[i], 8);
    }
}

#define ITERS 1000000

// Note: size is the *unpadded* size of the input vars
__global__ void fuzz(uint8_t *in_data, size_t size, const uint8_t *key, uint64_t *gobuf, unsigned long long *execs) {
  int bindex = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t rks[4 * 16] = {0};
  cham128_keygen(rks, key);
  float ldata;
  uint32_t blk0, blk1, blk2, blk3;
  blk0 = bindex;
  for (int i = 0; i < ITERS; i++) {
    // cham, one round
    const uint32_t* rk = (const uint32_t*) rks;
    uint32_t rc = 0;
    blk0 = rol32((blk0 ^ (rc++)) + (rol32(blk1, 1) ^ rk[0]), 8);
    blk1 = rol32((blk1 ^ (rc++)) + (rol32(blk2, 8) ^ rk[1]), 1);
    blk2 = rol32((blk2 ^ (rc++)) + (rol32(blk3, 1) ^ rk[2]), 8);
    blk3 = rol32((blk3 ^ (rc++)) + (rol32(blk0, 8) ^ rk[3]), 1);
    blk0 = rol32((blk0 ^ (rc++)) + (rol32(blk1, 1) ^ rk[4]), 8);
    blk1 = rol32((blk1 ^ (rc++)) + (rol32(blk2, 8) ^ rk[5]), 1);
    blk2 = rol32((blk2 ^ (rc++)) + (rol32(blk3, 1) ^ rk[6]), 8);
    blk3 = rol32((blk3 ^ (rc++)) + (rol32(blk0, 8) ^ rk[7]), 1);

    memcpy(&ldata, &blk0, sizeof(ldata));
    LLVMFuzzerTestOneInput(ldata, size);
  }
  return;
}

void CUDART_CB finishedCB(void *data) {
  finished_dev = *(int *)data;
}

void launch_kernel(int device, int varsize, uint8_t **ret_gbuf, uint64_t **ret_gobuf, unsigned long long **ret_execs) {
  cudaSetDevice(device);

  uint64_t *gobuf;
  unsigned long long *gexecs;

  unsigned char ckey[AES_BLOCK_SIZE];
  FILE *rng = fopen("/dev/urandom","rb");
  fread(ckey, AES_BLOCK_SIZE, 1, rng);
  fclose(rng);

  const uint8_t *dkey;
  gpuErrchk(cudaMalloc(&dkey, 16));
  gpuErrchk(cudaMemcpy((uint8_t *)dkey, ckey, 16, cudaMemcpyHostToDevice));

  // Alloc GPU buffers
  gpuErrchk(cudaMalloc(&gobuf, sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&gexecs, sizeof(unsigned long long)));

  *ret_gobuf = gobuf;
  *ret_execs = gexecs;

  // Start fuzzing!
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreate(&stream));
  int *dev = (int *)malloc(sizeof(int));
  *dev = device + 1;
  printf("Launching kernel on GPU%d...\n", device);
  fuzz<<<M,N,0,stream>>>(NULL, varsize, dkey, gobuf, gexecs);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaLaunchHostFunc(stream, finishedCB, dev));
}

int main(int argc, char **argv) {
  int NUM_GPU;
  gpuErrchk( cudaGetDeviceCount(&NUM_GPU) );
  if (NUM_GPU < 1) {
    fprintf(stderr, "No CUDA-capable GPUs detected!\n");
    return 1;
  }

  uint8_t *gbuf[NUM_GPU];
  uint64_t *gobuf[NUM_GPU];
  unsigned long long *goexecs[NUM_GPU];

  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
  for (int i = 0; i < NUM_GPU; i++) {
    launch_kernel(i, varsize, &gbuf[i], &gobuf[i], &goexecs[i]);
  }

  printf("Waiting on GPUs...\n");
  // Wait to finish
  for (int i = 0; i < NUM_GPU; i++) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
  }
  unsigned long long hexecs = (unsigned long long)N * (unsigned long long)M * (unsigned long long)ITERS * (unsigned long long)NUM_GPU;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double seconds = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + (end.tv_sec  - begin.tv_sec);
  printf("Did %llu execs in %f seconds, %f execs/s\n", hexecs, seconds, hexecs / seconds);
}
