#include <cstdio>

int main() {
  int a = 0;
#pragma omp parallel for reduction(+:a)
  for(int i=0; i<10000; i++) {
    a += 1;
    // printf("%d ",a);
  }
  printf("%d\n",a);
}

// 各スレッドでは途中までしか計算していない
// が，最終的には 10000 が出力される

// atomic よりも良い実装
// いい感じにやってくれる
