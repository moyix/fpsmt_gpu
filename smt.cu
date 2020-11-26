#include <time.h>
#include <sys/time.h>
#include <sched.h>
#include "SMTLIB/Float.h"
#include "SMTLIB/BufferRef.h"
#include "cuda_aes.h"
#include "theory.h"
#include "smt.h"

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


// Note: size is the *unpadded* size of the input vars
__global__ void fuzz(uint8_t *in_data, size_t size, const uint8_t *key, uint64_t *gobuf) {
  int bindex = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t padded = aes_pad(size);
  uint64_t offset = bindex * padded;

  // Get our local chunk
  const uint8_t *data = in_data + offset;

  // First time initialize block to i
  for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
    *(uint64_t *)(data+i) = bindex * (padded/AES_BLOCK_SIZE) + i;
  }

  while (!solved) {
    // Randomize input for our slice
    for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
      encrypt_one_table(in_data, key, offset+i);
    }
    if (LLVMFuzzerTestOneInput(data, size)) {
      *gobuf = bindex;
      solved = 1;
    }
  }
  return;
}

void CUDART_CB finishedCB(void *data) {
  finished_dev = *(int *)data;
}

void launch_kernel(int device, int varsize, uint8_t **ret_gbuf, uint64_t **ret_gobuf) {
  cudaSetDevice(device);

  uint8_t *gbuf;
  uint64_t *gobuf;

  int64_t padded = aes_pad(varsize);

  unsigned char ckey[AES_BLOCK_SIZE];
  FILE *rng = fopen("/dev/urandom","rb");
  fread(ckey, AES_BLOCK_SIZE, 1, rng);
  fclose(rng);

  // Pre-expand the round keys and copy to device mem
  uint8_t rkey[176];
  const uint8_t *drkey;
  expand_key(ckey, rkey);
  gpuErrchk(cudaMalloc(&drkey, sizeof(uint8_t) * 176));
  gpuErrchk(cudaMemcpy((uint8_t *)drkey, rkey, sizeof(uint8_t) * 176, cudaMemcpyHostToDevice));

  // Alloc GPU buffers
  gpuErrchk(cudaMalloc(&gbuf, padded*N*M*sizeof(uint8_t)));
  gpuErrchk(cudaMalloc(&gobuf, sizeof(uint64_t)));

  *ret_gbuf = gbuf;
  *ret_gobuf = gobuf;

  // Start fuzzing!
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreate(&stream));
  int *dev = (int *)malloc(sizeof(int));
  *dev = device + 1;
  printf("Launching kernel on GPU%d...\n", device);
  fuzz<<<M,N,0,stream>>>(gbuf, varsize, drkey, gobuf);
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

  for (int i = 0; i < NUM_GPU; i++) {
    launch_kernel(i, varsize, &gbuf[i], &gobuf[i]);
  }

  printf("Waiting on GPUs...\n");
  while (!finished_dev) sched_yield();
  int i = finished_dev - 1;
  printf("Search completed on device %d\n", i);

  // Get and print output
  int64_t padded = aes_pad(varsize);
  uint8_t *buf = (uint8_t *)malloc(padded);
  uint64_t oindex;
  gpuErrchk(cudaMemcpy(&oindex, gobuf[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buf, gbuf[i]+(oindex*padded), padded, cudaMemcpyDeviceToHost));
  printf("Found a satisfying assignment on device %d thread %lu:\n", i, oindex);
  for (int k = 0; k < varsize; k++) printf("%02x", buf[k]); printf("\n");
}
