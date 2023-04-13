#include <cstdio>

int main() {
  int a = 5, b = 5, c = 5;
#pragma omp parallel num_threads(4)
  {
#pragma omp for private(a)
    for(int i=0; i<4; i++)
      printf("%d ",++a);
#pragma omp single
    printf("\n");
#pragma omp for firstprivate(b)
    for(int i=0; i<4; i++)
      printf("%d ",++b);
#pragma omp single
    printf("\n");
#pragma omp for lastprivate(c)
    for(int i=0; i<4; i++)
      printf("%d ",++c);
#pragma omp single
    printf("\n");
  }
  printf("%d %d %d\n",a,b,c);
}

// private の種類について

// private : 1 1 1 1 (last : 5)
// copy in  : しない
// copy out : しない

// firstprivate : 6 6 6 6 (last : 5)
// copy in  : する
// copy out : しない

// lastprivate : 1 1 1 1 (last : 1)
// copy in  : しない
// copy out : する

// private の場合は未初期化値も出力された
