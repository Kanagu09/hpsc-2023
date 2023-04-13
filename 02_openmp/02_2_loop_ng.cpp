#include <cstdio>
#include <omp.h>

#define OUT_NUM 16

// Bad pattern
// 前の反復に依存するような計算を並列化してみる

int main() {
  int in[OUT_NUM];
  for(int i=0; i<OUT_NUM; i++) {
    in[i] = 1;
  }

#pragma omp parallel for
  for(int i=0; i<OUT_NUM-1; i++) {
    printf("%d: %d\n",omp_get_thread_num(),i);
    in[i] = in[i] + in[i+1];
  }

  for(int i=0; i<OUT_NUM; i++) {
    printf("%d ",in[i]);
  }
  printf("\n");
}

// 計算結果がめちゃくちゃになった
