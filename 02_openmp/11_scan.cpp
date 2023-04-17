#include <cstdio>
#include <cstdlib>

// serial prefix sum
// int main(){
//   // init
//   const int N=8;
//   int a[N], b[N];
//   for(int i=0; i<N; i++){
//     a[i] = rand() & 3;
//     printf("%*d ",2,a[i]);
//   }
//   printf("\n");

//   for(int i=1;i<N;i++)
//     a[i] += a[i-1];

//   // out
//   for(int i=0; i<N; i++){
//     printf("%*d ",2,a[i]);
//   }
//   printf("\n");
// }

// parallel prefix sum
int main() {
  // init
  const int N=8;
  int a[N], b[N];
  for(int i=0; i<N; i++) {
    a[i] = rand() & 3;
    printf("%*d ",2,a[i]);
  }
  printf("\n");

#pragma omp parallel
  for(int j=1; j<N; j<<=1) {
#pragma omp for
    for(int i=0; i<N; i++)
      b[i] = a[i];
#pragma omp for
    for(int i=j; i<N; i++)
      a[i] += b[i-j];
  }

  // out
  for(int i=0; i<N; i++)
    printf("%*d ",2,a[i]);
  printf("\n");
}

// 1. 1つ前を加算する
// 2. その後，2つ前を加算する
// 3. さらに，4つ前を加算する
// によって，3段階で加算していけるように処理を分割している
