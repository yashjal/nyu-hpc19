#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream.h>

long min(long a,long b) {
  if (a<b) {
    return a;
  }
  return b;
}

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  long nthreads, chunk_size;
  
  #pragma omp parallel
  {
    long tid = omp_get_thread_num();
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      chunk_size = ceil(nthreads/n);
    }
    std::cout << tid << std::endl;

    #pragma omp barrier
    for (long i = tid*chunk_size; i<min(tid*chunk_size+chunk_size,n); i++) {
      prefix_sum[i] += A[i];
    }
  }

  //#pragma omp parallel for schedule(static,chunk_size)
  //for (long i=omp_get_thread_num()*chunk_size; i<(omp_get_thread_num()*chunk_size+chunk_size); i++) {
    //prefix_sum[i] += A[i];
  //}

  //printf("here");
  for (long i=chunk_size; i<n; i+= chunk_size) {
    long tmp = prefix_sum[i-1];
    for (long j = i; j < chunk_size; j++) {
      prefix_sum[j] += tmp;
    }
  }

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) { 
    A[i] = rand();
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
