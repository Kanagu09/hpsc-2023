#include <cstdio>

__device__ __managed__ int sum;

__global__ void reduction(int &sum) {
  //sum += 1;
  atomicAdd(&sum, 1);
}

int main(void) {
  const int N = 128;
  const int M = 64;
  sum = 0;
  reduction<<<N/M,M>>>(sum);
  cudaDeviceSynchronize();
  printf("%d\n",sum);
}

// atomicAdd は，大量の thread から同時に global memory にアクセスが飛ぶので，thread 数によってはかなり処理が遅くなる

