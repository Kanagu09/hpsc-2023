#include <cstdio>

__global__ void thread(float *a) {
  a[threadIdx.x] = threadIdx.x;
}

int main(void) {
  const int N = 4;
  float *a;
  cudaMallocManaged(&a, N*sizeof(float));
  thread<<<1,N>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  cudaFree(a);
}

// share mamory

// N: max 1024
// block あたりの thread 数は1024が最大

// block 数の最大は 2048

