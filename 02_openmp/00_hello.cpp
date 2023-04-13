#include <iostream>

int main() {
#pragma omp parallel
  std::cout << "hello" << std::endl;
  // std::cout << "hello\n";
}

// helloと改行がめちゃくちゃになって出力される
// "\n"を使った方はきちんと1行ずつになる
