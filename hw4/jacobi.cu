#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define BLOCK_DIM 32

double norm(double *err, long N) {
  double sum = 0;
  //#pragma omp parallel for reduction (+:sum)
  for (long i = 0; i < N*N ; i+=1) {
     sum += err[i]*err[i];
  }
  return sqrt(sum);
}

__global__ void jacobi_kernel_nsmem(double* u, double* f, double * err, double * temp, long N, double h){
 int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x + 1;
 int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y + 1;
 temp[idx*N+idy] = (h*h*f[idx*N+idy] + u[(idx-1)*N+idy] + u[idx*N+idy-1] + u[(idx+1)*N+idy] + u[idx*N+idy+1])/4;
 __syncthreads();
 u[idx*N + idy] = temp[idx*N+idy];
 err[idx*N+idy] = (-u[(idx-1)*N+idy] - u[idx*N+idy-1] + 4*u[idx*N+idy] - u[(idx+1)*N+idy] - u[idx*N+idy+1] )/(h*h) - f[idx*N+idy];
}

int main() {
  long N = 3202; // dimension of 2D space
  double h = 1.0/(N+1); // size of update step
  double *u, *f, *err; // u is the solution, f is the forcing function

  cudaMallocHost((void**)&u, (N)*(N)*sizeof(double));
  cudaMallocHost((void**)&f, (N)*(N)*sizeof(double));
  cudaMallocHost((void**)&err, N*N*sizeof(double));

  for(long i = 0; i < N; i++) {
    for(long j = 0; j < N; j++) {
      u[i*N+j] = 0;
      f[i*N+j] = 1;
      err[i*N+j] = 0;
    }
  }

  dim3 blockDim(BLOCK_DIM,BLOCK_DIM);
  dim3 gridDim( (N-2)/BLOCK_DIM,(N-2)/BLOCK_DIM);
  double *u_d, *f_d, *temp_d, *err_d;
  cudaMalloc(&u_d, N*N*sizeof(double));
  cudaMalloc(&f_d, N*N*sizeof(double));
  cudaMalloc(&temp_d, N*N*sizeof(double));
  cudaMalloc(&err_d, N*N*sizeof(double));

  cudaMemcpyAsync(u_d, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(f_d, f, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  long t = 1;
  double tt2 = 0;
  double tt = omp_get_wtime();
  while (t < 10000) {
    jacobi_kernel_nsmem<<<gridDim, blockDim, 0>>>(u_d,f_d,err_d, temp_d, N, h );
    if ((t % 1000) == 0) {
      tt2 += omp_get_wtime()-tt;
      //printf("time = %f\n", omp_get_wtime()-tt );
      //printf("Bandwidth = %f\n", 1000*7*(N-2)*(N-2)/(omp_get_wtime()-tt)/1e9);
      cudaMemcpyAsync(err,err_d,N*N*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      printf("Iters = %d, Err = %f\n", t, norm(err, N));
      tt = omp_get_wtime();
    }
    t += 1;
  }
  printf("total GPU time = %f\n", tt2);
  cudaFree(u_d);
  cudaFree(f_d);
  cudaFree(temp_d);
  cudaFree(err_d);
  cudaFreeHost(u);
  cudaFreeHost(f);
  cudaFreeHost(err);
  return 0;
}

