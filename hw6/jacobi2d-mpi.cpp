#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i, j;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lN; i++) {
    for (j = 1; j <= lN; j++) {
      tmp = invhsq * (-lu[(i-1)*(lN+2)+j] - lu[i*(lN+2)+(j-1)] + 4*lu[i*(lN+2)+j] - lu[(i+1)*(lN+2)+j] - lu[i*(lN+2)+(j+1)]) - 1;
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, j, jp, p, N, lN, iter, max_iters;
  MPI_Status status, status1, status2, status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p); // assuming p is 4^j

  jp = (int)(log2(p)/2);
  int j2 = pow(2,jp);
  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N/j2;
  if ((N % j2 != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of 2^j\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lutemp;
  double * lu_col_send = (double *) calloc(sizeof(double), lN);
  double * lu_col_recv = (double *) calloc(sizeof(double), lN);

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;
  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
      for (j = 1; j <= lN; j++) {
        lunew[i*(lN+2)+j]  = 0.25 * (hsq + lu[(i-1)*(lN+2)+j] + lu[i*(lN+2)+(j-1)] + lu[(i+1)*(lN+2)+j] + lu[i*(lN+2)+(j+1)]);
      }
    }
    /* communicate ghost values */
    if (mpirank > j2-1) {
      /* If not the last process, send/recv bdry values to the right */
      MPI_Send(&(lunew[lN+3]), lN, MPI_DOUBLE, mpirank-j2, 124, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1]), lN, MPI_DOUBLE, mpirank-j2, 123, MPI_COMM_WORLD, &status);
    }
    if (mpirank < p-j2) {
      /* If not the first process, send/recv bdry values to the left */
      MPI_Send(&(lunew[(lN+2)*(lN)+1]), lN, MPI_DOUBLE, mpirank+j2, 123, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[(lN+2)*(lN+1)+1]), lN, MPI_DOUBLE, mpirank+j2, 124, MPI_COMM_WORLD, &status1);
    }
    // removes right col
    if ((mpirank+1)%j2 != 0) {
      for (i = 0; i < lN; i++) {
        lu_col_send[i] = lunew[(i+1)*(lN+2)+lN];
      }
      MPI_Send(lu_col_send, lN, MPI_DOUBLE, mpirank+1, 121, MPI_COMM_WORLD);
      MPI_Recv(lu_col_recv, lN, MPI_DOUBLE, mpirank+1, 122, MPI_COMM_WORLD, &status2);
      for (i = 0; i < lN; i++) {
        lunew[(i+1)*(lN+2)+lN+1] = lu_col_recv[i];
      }
    }
    //removes left col
    if (mpirank%j2 != 0) {
      for (i = 0; i < lN; i++) {
        lu_col_send[i] = lunew[(i+1)*(lN+2)+1];
      }
      MPI_Send(lu_col_send, lN, MPI_DOUBLE, mpirank-1, 122, MPI_COMM_WORLD);
      MPI_Recv(lu_col_recv, lN, MPI_DOUBLE, mpirank-1, 121, MPI_COMM_WORLD, &status3);
      for (i = 0; i < lN; i++) {
        lunew[(i+1)*(lN+2)] = lu_col_recv[i];
      }
    }

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(lu_col_send);
  free(lu_col_recv);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
