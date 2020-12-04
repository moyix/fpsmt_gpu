#include <time.h>
#include <sys/time.h>
#include <sched.h>
#include "curand_kernel.h"
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
//__global__ void fuzz(uint8_t *in_data, size_t size, const uint8_t *key, uint64_t *gobuf, unsigned long long *execs) {
__global__ void fuzz(uint8_t *in_data, size_t size, curandState *state, uint64_t *gobuf, unsigned long long *execs) {
  int bindex = blockIdx.x * blockDim.x + threadIdx.x;

  uint8_t *data = in_data + bindex*size; // i think?

  curandState localState = state[bindex];

  while (!solved) {
    atomicAdd(execs, 1);
    // Randomize input for our slice
    uint8_t* curr = data;
    //TODO: once we confirm 16bytes and we generate 8bytes, replace loop with writes
    while (curr < data + size)
    {
        *curr++ = curand(&localState); //TODO: i think this is 8bytes but not sure, alternative is uint4
    }

    if (LLVMFuzzerTestOneInput(data, size)) {
      *gobuf = bindex;
      memcpy(in_data+offset, sdata+soff, size);
      solved = 1;
    }
    // Add increment to randomize (I hope?)
    for (int i = 0; i < padded; i += AES_BLOCK_SIZE) {
      *(uint64_t *)(sdata+soff+i) = bindex * (padded/AES_BLOCK_SIZE) + i;
    }
  }
  return;
}

void CUDART_CB finishedCB(void *data) {
  finished_dev = *(int *)data;
}

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int seed = id*37;
    curand_init(seed, id, 0, &state[id]);
}

void launch_kernel(int device, int varsize, uint8_t **ret_gbuf, uint64_t **ret_gobuf, unsigned long long **ret_execs) {
  cudaSetDevice(device);

  uint8_t *gbuf;
  uint64_t *gobuf;
  unsigned long long *gexecs;

  int size = varsize; // i think?
  curandState *rngStates;
  gpuErrchk(cudaMalloc(&rngStates, N*M*sizeof(curandState)));

  setup_kernel<<<M,N>>>(rngStates);

  // Alloc GPU buffers
  gpuErrchk(cudaMalloc(&gbuf, size*N*M));
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
  //fuzz<<<M,N,0,stream>>>(gbuf, varsize, rnd, gobuf, gexecs);
  fuzz<<<M,N,0,stream>>>(gbuf, varsize, rngStates, gobuf, gexecs);
  gpuErrchk(cudaLaunchHostFunc(stream, finishedCB, dev));
  //gpuErrchk(curandDestroyGenerator(gen));
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
  while (!finished_dev) sched_yield();
  int i = finished_dev - 1;
  // Wait to finish
  cudaSetDevice(i);
  cudaDeviceSynchronize();
  unsigned long long hexecs;
  printf("Search completed on device %d\n", i);
  gpuErrchk(cudaMemcpy(&hexecs, goexecs[i], sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  float seconds = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + (end.tv_sec  - begin.tv_sec);
  printf("Did %llu execs in %f seconds, %f execs/s\n", hexecs, seconds, hexecs / seconds);


  // Get and print output
  int64_t padded = aes_pad(varsize);
  uint8_t *buf = (uint8_t *)malloc(padded);
  uint64_t oindex;
  gpuErrchk(cudaMemcpy(&oindex, gobuf[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buf, gbuf[i]+(oindex*padded), padded, cudaMemcpyDeviceToHost));
  printf("Found a satisfying assignment on device %d thread %lu:\n", i, oindex);
  for (int k = 0; k < varsize; k++) printf("%02x", buf[k]); printf("\n");
}
