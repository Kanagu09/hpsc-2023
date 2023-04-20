#include <cstdio>
#include <immintrin.h>

int main() {
  const int N = 8;
  float a[N];
  for (int i=0; i<N; i++)
    a[i] = 1;
  __m256 avec = _mm256_load_ps(a);
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  _mm256_store_ps(a, bvec);
  for (int i=0; i<N; i++)
    printf("%g\n",a[i]);
}

// hadd で足す部分が変な形になっているので， permute で順序を入れ替えている
// a : 01234567
// b : 45670123
