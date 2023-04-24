#include <cstdio>
#include <openacc.h>

int main() {
#pragma acc parallel loop num_gangs(2) num_workers(2) vector_length(2)
  for(int i=0; i<8; i++) {
    printf("(%d,%d,%d): %d\n",
           __pgi_gangidx(),
           __pgi_workeridx(),
           __pgi_vectoridx(),i);
  }
}

// gang, worker, vector の数を指定できる
// block, warp, thread に該当

// どの数で高い性能が出るかを探る
// →チューニング

