#include <iostream>
#include <omp.h>

int main() {
  omp_set_num_threads(3);
#pragma omp parallel num_threads(2)
  std::cout << "hello\n";
}

// 並列の個数を指定できる
// 優先度は以下
// 1. ディレクティブ num_threads()
// 2. 関数 omp_set_num_threads()
// 3. 環境変数OMP_NUM_THREADS (実行時指定)
