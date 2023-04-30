#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int *vec){
  vec[threadIdx.x] = 0;
}

__global__ void copy(int *vec1, int *vec2){
  vec1[threadIdx.x] = vec2[threadIdx.x];
}

__global__ void count(int *bucket, int *key){
  atomicAdd(&bucket[key[threadIdx.x]], 1);
}

__global__ void set_offset(int range, int *offset, int *buf){
  int i = threadIdx.x;
  for(int j=1; j<range; j<<=1) {
    buf[i] = offset[i];
    if (i >= j)
      offset[i] += buf[i-j];
  }
}

__global__ void set_key(int range, int *key, int *offset){
  for(int j=1; j<range; j++){
    if(offset[j-1] <= threadIdx.x)
      key[threadIdx.x] = j;
  }
}

int main() {
  int n = 50;
  int range = 5;

  int *bucket;
  int *key;
  int *offset;
  int *buf;
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
  cudaMallocManaged(&buf, range*sizeof(int));

  // init
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // sort
  init<<<1,range>>>(bucket);
  init<<<1,range>>>(offset);
  cudaDeviceSynchronize();

  count<<<1, n>>>(bucket, key);
  cudaDeviceSynchronize();

  copy<<<1,range>>>(offset, bucket);
  cudaDeviceSynchronize();

  set_offset<<<1,range>>>(range, offset, buf);
  cudaDeviceSynchronize();

  init<<<1,n>>>(key);
  cudaDeviceSynchronize();

  set_key<<<1,n>>>(range, key, offset);
  cudaDeviceSynchronize();

  // print
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  // free
  cudaFree(bucket);
  cudaFree(key);
  cudaFree(offset);
  cudaFree(buf);
}
