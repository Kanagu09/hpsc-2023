#include <cstdio>
#include <omp.h>

#define OUT_NUM 16

// Good pattern
// 前の反復に依存しないように書き換える

int main() {
  int in[OUT_NUM];
  int out[OUT_NUM];
  for(int i=0; i<OUT_NUM; i++) {
    in[i] = 1;
    out[i] = 1;
  }

#pragma omp parallel for
  for(int i=0; i<OUT_NUM-1; i++) {
    printf("%d: %d\n",omp_get_thread_num(),i);
    out[i] = in[i] + in[i+1];
  }

#pragma omp parallel for
  for(int i=0; i<OUT_NUM; i++) {
    in[i] = out[i];
  }

  for(int i=0; i<OUT_NUM; i++) {
    printf("%d ",out[i]);
  }
  printf("\n");
}

// 計算結果は正しくなった
