#include <cstdio>

__global__ void block(float *a, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=N) return;
  a[i] = i;
}

int main(void) {
  const int N = 2000;
  const int M = 1024;
  float *a;
  cudaMallocManaged(&a, N*sizeof(float));
  block<<<(N+M-1)/M,M>>>(a,N);
  cudaDeviceSynchronize();
  for (int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  cudaFree(a);
}

// (N+M-1)/M : N/Mの切り上げ
// ( N/M は切り捨てなので )

// thread で足りない分は， block を用いることで処理できる

