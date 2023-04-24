#include <cstdio>
#include <vector>

int main() {
  std::vector<int> a(8);
#pragma acc parallel loop
  for(int i=0; i<8; i++)
    a[i] = i;
  for(int i=0; i<8; i++)
    printf("%d\n",a[i]);
}

// -gpu=managed をつけないと，実行時にエラー

// OpenACC は header まで読みに行く

