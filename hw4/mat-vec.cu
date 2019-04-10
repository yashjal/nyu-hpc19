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

// Warp divergence
__global__ void reduction_kernel0(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x %   2 == 0) smem[threadIdx.x] += smem[threadIdx.x + 1];
  __syncthreads();
  if (threadIdx.x %   4 == 0) smem[threadIdx.x] += smem[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x %   8 == 0) smem[threadIdx.x] += smem[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x %  16 == 0) smem[threadIdx.x] += smem[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x %  32 == 0) smem[threadIdx.x] += smem[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x %  64 == 0) smem[threadIdx.x] += smem[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x % 128 == 0) smem[threadIdx.x] += smem[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x % 256 == 0) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x % 512 == 0) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + 512];
}

// Shared memory bank conflicts
__global__ void reduction_kernel1(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x *   2] += smem[threadIdx.x *   2 +   1];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x *   4] += smem[threadIdx.x *   4 +   2];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x *   8] += smem[threadIdx.x *   8 +   4];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x *  16] += smem[threadIdx.x *  16 +   8];
  __syncthreads();
  if (threadIdx.x <  32) smem[threadIdx.x *  32] += smem[threadIdx.x *  32 +  16];
  __syncwarp();
  if (threadIdx.x <  16) smem[threadIdx.x *  64] += smem[threadIdx.x *  64 +  32];
  __syncwarp();
  if (threadIdx.x <   8) smem[threadIdx.x * 128] += smem[threadIdx.x * 128 +  64];
  __syncwarp();
  if (threadIdx.x <   4) smem[threadIdx.x * 256] += smem[threadIdx.x * 256 + 128];
  __syncwarp();
  if (threadIdx.x <   2) smem[threadIdx.x * 512] += smem[threadIdx.x * 512 + 256];
  __syncwarp();
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[512];
}

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

__global__ void vec_mult_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] * b[idx];
}


int main() {
  long N = (1UL<<25);

  double *x;
  double *y;
  double *z;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&z, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = 1.0;
  }

  double sum_ref, sum;
  double tt = omp_get_wtime();
  for (long i=0; i<N;i++) {
    z[i] = x[i] * y[i];
  }
  reduction(&sum_ref, z, N);
  printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *z_d, *c_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, N*sizeof(double));
  cudaMalloc(&c_d, ((N+BLOCK_SIZE-1)/BLOCK_SIZE)*sizeof(double));

  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  vec_mult_kernel<<<N/BLOCK_SIZE+1,BLOCK_SIZE>>>(z_d, y_d, x_d, N);
  cudaDeviceSynchronize();
  double* sum_d = c_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d, z_d, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + Nb, sum_d, N);
    sum_d += Nb;
  }

  cudaMemcpyAsync(&sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %f\n", fabs(sum-sum_ref));
  printf("sum %f\n", sum);
  printf("sum_ref %f\n", sum_ref);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(c_d);
  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(z);

  return 0;
}
