#include <cstdio>

int main() {
  int a = 0;
#pragma acc parallel loop reduction(+:a)
  for(int i=0; i<10000; i++) {
    a += 1;
  }
  printf("%d\n",a);
}

// reduction をつけないと，並列化されない
// (OpenMPは，挙動おかしくなりながらも並列化する)

