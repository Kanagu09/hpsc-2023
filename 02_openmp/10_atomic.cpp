#include <cstdio>

int main() {
  float x[10];
  for (int i=0; i<10; i++)
    x[i] = 0.0;
#pragma omp parallel for
  for (int i=0; i<1000; i++) {
#pragma omp atomic update
    x[i%10]++;
    // printf("%g ",x[i%10]);
  }
  for (int i=0; i<10; i++)
    printf("%d %g\n",i,x[i]);
}

// 各スレッドの段階で100まで計算されている

// lock してくれる
// ただし，処理は遅くなる

// #pragma omp critical
// #pragma omp single
// を入れて，無理やり干渉しないように制御する方法もある
// (ただし処理は遅い)
