#include <cstdio>
#include <omp.h>

int main() {
  int a = 0;
  int b[1] = {0};
#pragma omp parallel for
  for(int i=0; i<8; i++) {
    a = omp_get_thread_num();
  // #pragma omp parallel private(i)
    b[0] = omp_get_thread_num();
  }
  printf("%d %d\n",a,b[0]);
}

// 並列化よりも前に宣言されているとshared
// 並列化よりも後に宣言されているとprivate
// #pragma omp parallel private(i) を入れると，privateにできる

// a, bはsharedなので，いずれかのスレッドで上書きされた値が出力される
// privateにする行を入れた場合は，privateになるので出力は 0
