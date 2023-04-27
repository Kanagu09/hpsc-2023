#include <cstdio>

__device__ __managed__ int sum;

__global__ void reduction(int &sum, int *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int b[];
  __syncthreads();
  b[threadIdx.x] = a[i];
  __syncthreads();
  int c = 0;
  for (int j=0; j<blockDim.x; j++)
    c += b[j];
  if (threadIdx.x == 0)
    atomicAdd(&sum, c);
}

int main(void) {
  const int N = 128;
  const int M = 64;
  int *a;
  cudaMallocManaged(&a, N*sizeof(int));
  for (int i=0; i<N; i++) a[i] = i;
  reduction<<<N/M,M,M*sizeof(int)>>>(sum, a);
  cudaDeviceSynchronize();
  printf("%d\n",sum);
  cudaFree(a);
}

// shared memory を使うと， L1 相当で load できる
// block 内の足し込みが高速化

// extern __shared__ int b[];
// は
// __shared__ int b[64];
// に相当

// foo<<<block, thread, shard memory>>>();
// shared memory の動的確保

