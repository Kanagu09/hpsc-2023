#include <cstdio>

// __attribute__ ((noinline)) void add(float a, float b, float &c) {
void add(float a, float b, float &c) {
  c = a + b;
}

int main() {
  const int N = 8;
  float a[N], b[N], c[N];
  for(int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i * 0.1;
    c[i] = 0;
  }
  for(int i=0; i<N; i++)
    add(a[i],b[i],c[i]);
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,c[i]);
}

// インライン化できる関数ならば，関数もベクトル化できる

// __attribute__ ((noinline)) でインライン化を禁止すると，ベクトル化はされない
