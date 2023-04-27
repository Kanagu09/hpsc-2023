#include <cstdio>

__global__ void oddeven(float *a) {
  int i = threadIdx.x;
  if (i & 1)
    a[i] = -i;
  else
    a[i] = i;
}

int main(void) {
  const int N = 32;
  float *a;
  cudaMallocManaged(&a, N*sizeof(float));
  oddeven<<<1,N>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  cudaFree(a);
}

// if 文は，真と偽両方を計算しておいて，後から mask で処理している
// -> 計算資源としては節約できない

