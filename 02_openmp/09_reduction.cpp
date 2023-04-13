#include <cstdio>

int main() {
  int a = 0;
#pragma omp parallel for reduction(+:a)
  for(int i=0; i<10000; i++) {
    // printf("%d ",a);
    a += 1;
  }
  printf("%d\n",a);
}

// 各スレッドでは途中までしか計算していない
// が，最終的には 10000 が出力される
