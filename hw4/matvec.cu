#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void reduction(double* sum_ptr, const double* a, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i];
  *sum_ptr = sum;
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void multiply(double* a, double* b, double* c, long N){
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  if(idx < N) c[idx] = a[idx]*b[idx];
}

int main() {
  long N = (1UL<<16);

  double *A;
  double *y;
  double *z;
  double *matvec;
  cudaMallocHost((void**)&A, N* N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&z, N * sizeof(double));
  cudaMallocHost((void**)&matvec, N*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
     y[i] = 1.0/(i+1);
     for(long j = 0; j < N; j++) {
         A[i*N + j] = 1.0;
     }
  }
  double sum_ref, sum;
  double tt = omp_get_wtime();

  #pragma omp parallel for schedule(static)
  for(long i = 0; i < N; i++) {
    for(long j = 0; j < N; j++) {
      z[j] = A[i*N + j] * y[j];
    }
    reduction(&sum_ref, z, N);
    matvec[i] = sum_ref;
  }

  printf("CPU Bandwidth = %f GB/s\n", 4*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *w_d, *y_d, *z_d, *matvec2;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&w_d, ((N+BLOCK_SIZE-1)/BLOCK_SIZE)*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, N*sizeof(double));
  cudaMallocHost((void**)&matvec2, N*sizeof(double));
  
  double tt2 = 0;
  //#pragma omp parallel for schedule(static)
  for(long i = 0; i < N; i++) {
    cudaMemcpyAsync(x_d, &A[i*N], N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpyAsync(z_d, z, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    double ttp = omp_get_wtime();
    double* mult_d = z_d;
    double* sum_d = w_d;
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    multiply<<<Nb, BLOCK_SIZE>>>(x_d,y_d,mult_d,N);
    reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d, mult_d, N);
    while (Nb > 1) {
      long N = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + Nb, sum_d, N);
      sum_d += Nb;
    }
    tt2 += omp_get_wtime()-ttp;
    cudaMemcpyAsync(&sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    matvec2[i]=sum;
  }

  printf("GPU Bandwidth = %f GB/s\n", 4*N*N*sizeof(double) / (tt2)/1e9);

  double error = 0;
  for(long i = 0; i<N;i++){
    error += fabs(matvec[i] - matvec2[i]);
  }

  printf("Error = %f\n", error);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(w_d);
  cudaFreeHost(matvec);
  cudaFreeHost(matvec2);
  cudaFreeHost(A);
  cudaFreeHost(y);
  cudaFreeHost(z);
  return 0;
}

