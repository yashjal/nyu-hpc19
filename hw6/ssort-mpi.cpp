// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);
  printf("p: %d\n",p);
  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* lsplit = (int*)malloc((p-1)*sizeof(int));

  for (int i = 0; i < (p-1); i++) {
    printf("i: %d,   ", i);
    lsplit[i] = vec[(int)((i+1)*N/p)];
    printf("lsplit %d\n", lsplit[i]);
  }
  //MPI_Barrier(MPI_COMM_WORLD);
  printf("lsplit first entry: %d", lsplit[0]);

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* splits;
  if (rank == 0) {
    splits = (int*)malloc((p-1)*p*sizeof(int));
  } 
  
  printf("rank: %d", rank); 
  MPI_Gather(lsplit, N, MPI_INT, splits, N, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    std::sort(splits,splits+(p-1)*p);
    printf("splits first entry: %d", splits[0]);
    for (int i = 0; i < (p-1); i++) {
      lsplit[i] = splits[(i+1)*p];
    }
  }

  MPI_Bcast(lsplit, p-1, MPI_INT, 0, MPI_COMM_WORLD);
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  // root process broadcasts splitters to all other processes
  //MPI_Bcast(s);
  
  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  int* scounts = (int *) malloc(sizeof(int)*p);
  int* sdispls = (int *) malloc(sizeof(int)*p);
  sdispls[0] = 0;
  
  for (int i = 0; i < (p-1); i++) {
    sdispls[i+1] = std::lower_bound(vec, vec+N, lsplit[i]) - vec;
    scounts[i] = sdispls[i+1]-sdispls[i];
  }
  scounts[0] += 1;
  scounts[p-1] = (N-1)-sdispls[p-1];
  
  int* counts = (int *) malloc(sizeof(int)*p*p);
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  MPI_Alltoall(scounts,p,MPI_INT,counts,p,MPI_INT,MPI_COMM_WORLD);
  int sum = 0;
  int * rcounts = (int *) malloc(sizeof(int)*p);
  int * rdispls = (int *) malloc(sizeof(int)*p);
  int * lcounts = (int *) malloc(sizeof(int)*p);
  rdispls[0] = 0;
  for (int i = 0; i < p; i++) {
    sum += counts[i*p+rank];
    rcounts[i] = counts[i*p+rank];
    lcounts[i] = counts[rank*p + i];
    if (i != 0){
       rdispls[i] = sum;
    }
  }
  int* recvbf = (int *) malloc(sizeof(int)*sum);
  MPI_Alltoallv(vec,lcounts,sdispls,MPI_INT,recvbf,rcounts,rdispls,MPI_INT,MPI_COMM_WORLD);
  // do a local sort of the received data
  std::sort(recvbf,recvbf+sum);
  // every process writes its result to a file
  for (int i = 0; i < sum; i++) {
    printf("rank: %d, iter: %d, val: %d\n", rank, i, recvbf[i]);
  }

  free(vec);
  free(lsplit);
  free(counts);
  free(scounts);
  free(sdispls);
  free(recvbf);
  free(lcounts);
  free(rcounts);
  free(rdispls);
  if (rank == 0) {
    free(splits);
  }
  MPI_Finalize();
  return 0;
}
