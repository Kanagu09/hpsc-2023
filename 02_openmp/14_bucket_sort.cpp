#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // calc bucket num
  std::vector<int> bucket(range,0);
#pragma omp parallel for
  for (int i=0; i<n; i++){
#pragma omp atomic update
    bucket[key[i]]++;
  }

  // calc offset
  std::vector<int> offset(range,0);
  std::vector<int> buf(range,0);
#pragma omp prallel for
  for (int i=0; i<range-1; i++) {
    offset[i+1] = bucket[i];
  }
#pragma omp parallel
  for(int j=1; j<range; j<<=1) {
#pragma omp for
    for(int i=0; i<range; i++)
      buf[i] = offset[i];
#pragma omp for
    for(int i=j; i<range; i++)
      offset[i] += buf[i-j];
  }

  // sort
#pragma omp parallel for
  for(int i=0; i<n; i++){
#pragma omp parallel for
    for(int j=0; j<range; j++){
      if(offset[j] <= i)
        key[i] = j;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
