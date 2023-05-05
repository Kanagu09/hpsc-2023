#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }

  __m256 x_vec = _mm256_load_ps(x);
  __m256 y_vec = _mm256_load_ps(y);
  __m256 m_vec = _mm256_load_ps(m);
  __m256 fx_vec = _mm256_load_ps(fx);
  __m256 fy_vec = _mm256_load_ps(fy);
  __m256 zero_vec = _mm256_set1_ps(0);

  for(int i=0; i<N; i++) {
    // load x[i], y[i]
    __m256 xi_vec = _mm256_set1_ps(x[i]);
    __m256 yi_vec = _mm256_set1_ps(y[i]);

    // float rx = x[i] - x[j];
    // float ry = y[i] - y[j];
    __m256 rx_vec = _mm256_sub_ps(xi_vec, x_vec);
    __m256 ry_vec = _mm256_sub_ps(yi_vec, y_vec);

    // float r = std::sqrt(rx * rx + ry * ry);
    __m256 r_to_2_vec = _mm256_add_ps(_mm256_mul_ps(rx_vec, rx_vec), _mm256_mul_ps(ry_vec, ry_vec));
    __m256 r_inv_vec = _mm256_rsqrt_ps(r_to_2_vec);
    __m256 r_inv_to_3_vec = _mm256_mul_ps(_mm256_mul_ps(r_inv_vec, r_inv_vec), r_inv_vec);

    // fx[i] -= rx * m[j] / (r * r * r);
    // fy[i] -= ry * m[j] / (r * r * r);
    __m256 fx_i_vec = _mm256_mul_ps(_mm256_mul_ps(rx_vec, m_vec), r_inv_to_3_vec);
    __m256 fy_i_vec = _mm256_mul_ps(_mm256_mul_ps(ry_vec, m_vec), r_inv_to_3_vec);

    // branch (i != j) -> 0
    __m256 i_vec = _mm256_set1_ps(i);
    __m256 j_vec = _mm256_load_ps(j);
    __m256 mask = _mm256_cmp_ps(i_vec, j_vec, _CMP_NEQ_OS);
    fx_i_vec = _mm256_blendv_ps(zero_vec, fx_i_vec, mask);
    fy_i_vec = _mm256_blendv_ps(zero_vec, fy_i_vec, mask);

    // sum fx
    __m256 fx_vec = _mm256_permute2f128_ps(fx_i_vec, fx_i_vec, 1);
    fx_vec = _mm256_add_ps(fx_vec, fx_i_vec);
    fx_vec = _mm256_hadd_ps(fx_vec, fx_vec);
    fx_vec = _mm256_hadd_ps(fx_vec, fx_vec);
    fx_vec = _mm256_sub_ps(zero_vec, fx_vec);
    float fx_tmp[N];
    _mm256_store_ps(fx_tmp, fx_vec);
    fx[i] = fx_tmp[0];

    // sum fy
    __m256 fy_vec = _mm256_permute2f128_ps(fy_i_vec, fy_i_vec, 1);
    fy_vec = _mm256_add_ps(fy_vec, fy_i_vec);
    fy_vec = _mm256_hadd_ps(fy_vec, fy_vec);
    fy_vec = _mm256_hadd_ps(fy_vec, fy_vec);
    fy_vec = _mm256_sub_ps(zero_vec, fy_vec);
    float fy_tmp[N];
    _mm256_store_ps(fy_tmp, fy_vec);
    fy[i] = fy_tmp[0];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
