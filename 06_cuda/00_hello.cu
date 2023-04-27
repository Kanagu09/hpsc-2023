#include <cstdio>

__global__ void print(void) {
  printf("Hello GPU\n");
}

int main() {
  printf("Hello CPU\n");
  print<<<2,4>>>();
  cudaDeviceSynchronize();
}

// foo<<<blocks, threads_per_block>>>();

// cudaDeviceSynchronize() はGPU関数の終了が保証されない
// 動作としては Hello GPU が出力されなくなる

// GPU 関数は投げるだけ投げて終わり

