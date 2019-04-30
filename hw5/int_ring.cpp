#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_ring(int Nproc, long Nrepeat, int Nsize, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  int* msg = (int*) malloc(sizeof(int)*Nsize);
  for (int i = 0; i < Nsize; i++) msg[i] = 0;

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    MPI_Status status;
    if (rank == 0) {
      MPI_Send(msg, Nsize, MPI_INT, rank+1, repeat, comm);
      MPI_Recv(msg, Nsize, MPI_INT, Nproc-1, repeat, comm, &status);
      for (int i=0; i < Nsize; i++) msg[i] += rank;
    } else {
      MPI_Recv(msg, Nsize, MPI_INT, rank-1, repeat, comm, &status);
      for (int i=0; i < Nsize; i++) msg[i] += rank;
      MPI_Send(msg, Nsize, MPI_INT, (rank+1)%Nproc, repeat, comm);
    }
  }
  tt = MPI_Wtime() - tt;
  if (rank == 0) {
    for (int i = 0; i < Nsize; i++) printf("index: %d, total: %d\n", i, msg[i]);
  }
  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 2) {
    printf("Usage: mpirun ./int_ring <Nrepeat>\n");
    abort();
  }
  long Nrepeat = atoi(argv[1]);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  int Nproc;
  MPI_Comm_size(comm, &Nproc);
  int Nsize = 500000;
  double tt = time_ring(Nproc, Nrepeat, Nsize, comm);
  if (!rank) printf("ring latency: %e ms\n", tt/Nrepeat * 1000/Nproc);
  if (!rank) printf("ring bandwidth: %e GB/s\n", (Nproc*sizeof(int)*Nsize*Nrepeat)/tt/1e9);

  MPI_Finalize();
}
