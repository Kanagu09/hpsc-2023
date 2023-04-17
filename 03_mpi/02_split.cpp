#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char** argv) {
  const int N = 20;
  double x[N], y[N], m[N], fx[N], fy[N];
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int begin = rank * (N / size);
  int end = (rank + 1) * (N / size);
  printf("%d %d %d\n",rank,begin,end);
  srand48(rank);

  // init
  for(int i=begin; i<end; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  MPI_Finalize();
}

// 素朴な並列化方法
// 配列の分担を明示的に指定
