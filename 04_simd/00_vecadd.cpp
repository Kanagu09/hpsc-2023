#include <cstdio>

int main() {
  const int N = 8;
  float a[N], b[N], c[N];
  for(int i=0; i<N; i++) { // vectorized
    a[i] = i;
    b[i] = i * 0.1;
    c[i] = 0;
  }
  //asm volatile ("# begin loop");
  for(int i=0; i<N; i++) // vectorized
    c[i] = a[i] + b[i];
  //asm volatile ("# end loop");
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,c[i]);
}

// このくらいの単純なループは O3 で自動で並列化してくれる
// 何が並列化されるかはコンパイラのバージョン依存

// fopt-info-vec-optimized は並列化箇所を教えてくれるオプション
// -match=native を入れると，新しいレジスタを使ってよろしくやってくれる
