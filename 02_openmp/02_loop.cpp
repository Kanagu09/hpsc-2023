#include <cstdio>
#include <omp.h>

#define OUT_NUM 16

int main() {
#pragma omp parallel for
// #pragma omp parallel for schedule(static)
// #pragma omp parallel for schedule(dynamic)
// #pragma omp parallel for schedule(guided)
// #pragma omp parallel for schedule(auto)
  for(int i=0; i<OUT_NUM; i++) {
    printf("%d: %d\n",omp_get_thread_num(),i);
  }
}

// ループを並列化できる
