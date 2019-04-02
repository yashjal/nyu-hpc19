#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

long min(long a, long b){
  if (a < b) {
    return a;
  }
  return b;
}

void scan_omp(long* prefix_sum, const long* A, long n) {

  const long num_threads = 4;
  const long chunk_size = n/num_threads;
  omp_set_num_threads(num_threads);
  #pragma omp parallel  
  { 
    int tid = omp_get_thread_num();
    scan_seq(&prefix_sum[tid*chunk_size],&A[tid*chunk_size],chunk_size);
    // last part of array done by thread 0
    if (tid == 0){
      scan_seq(&prefix_sum[num_threads*chunk_size],&A[num_threads*chunk_size],n-num_threads*chunk_size);
    }
  }
  // summing
  for(long i = 1; i < num_threads; i++){
    for(long j = 0; j < chunk_size; j++){
      prefix_sum[j + i*chunk_size] = prefix_sum[j + i*chunk_size] + prefix_sum[i*chunk_size-1];
    }
  }
  // last part of array
  for(long i = 0; i < n-num_threads*chunk_size; i++){
    prefix_sum[i+num_threads*chunk_size] = prefix_sum[i+num_threads*chunk_size] + prefix_sum[num_threads*chunk_size-1];
  }
}

int main() {
  long N = 1000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));

  for (long i = 0; i < N; i++) {
    A[i] = rand();
    B0[i] = 0;
    B1[i] = 0;
  }

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  
  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
