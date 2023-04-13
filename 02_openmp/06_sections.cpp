#include <cstdio>
#include <omp.h>

int main() {
  int i = 1;
#pragma omp parallel num_threads(2)
#pragma omp sections firstprivate(i)
// #pragma omp sections private(i)
  {
#pragma omp section
    printf("%d\n",++i);
#pragma omp section
    printf("%d\n",++i);
  }
}

// section 並列化の例

// 2つの section で同じ値が出力される 2,2
// firstprivate(i) は， i の複製をスレッドごとに複製し， private として持つ

// firstprivate(i) を private(i) に変更すると，複製しないため，出力は 1,1 となる
