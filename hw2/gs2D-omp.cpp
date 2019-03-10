#include <stdio.h>
#include <iostream>
#include <cmath>
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define n 100

long N;
double h;
const double err_tol = 1e-8;
const int max_iter = 10000;

double res_norm(double *f, double *Au){
	double sum_sq = 0.0;
	for (long i = 0; i < (N+2)*(N+2); i++) {
		sum_sq = sum_sq + (Au[i] - f[i])*(Au[i] - f[i]);
	}
	return sqrt(sum_sq);
}

double compute_error(double *u, double *f, double *Au){
	for (long i = 1; i < N+1; i++) {
		for (long j = 1; j < N+1; j++) {
			Au[i + j*(N+2)] = (-u[i-1 + j*(N+2)] - u[i + (j-1)*(N+2)] + 4*u[i + j*(N+2)] - u[i+1 + j*(N+2)] - u[i + (j+1)*(N+2)] )/(h*h);

		}
	} 
	return res_norm(f,Au);
}

double gauss_seidel(double *u, double *f, double *temp, double *Au) {
	int iter = 0;
        double error = 1e10;
	while ((iter < max_iter) && (error > err_tol)) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (long i = 1; i < N+1; i++) {
			for (long j = 1; j < N+1; j++) {
				if ((i + j) % 2 == 0) {
					temp[i + j*(N+2)] = (f[i + j*(N+2)]*h*h + u[i-1 + j*(N+2)] + u[i + (j-1)*(N+2)] + u[i+1 + j*(N+2)] + u[i + (j+1)*(N+2)])/4;
				}
			}
		}
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (long i = 1; i < N+1; i++ ) {
			for (long j = 1; j < N+1; j++ ) {
				if ((i+j) % 2 == 1){
					temp[i + j*(N+2)] = (f[i + j*(N+2)]*h*h + temp[i-1 + j*(N+2)] + temp[i + (j-1)*(N+2)] + temp[i+1 + j*(N+2)] + temp[i + (j+1)*(N+2)])/4;
				}
			}
		}
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (long i = 1; i < N+1; i++) {
			for(long j = 1; j < N+1; j++) {
				u[i + j*(N+2)] = temp[i + j*(N+2)];
			}
		}
		if (iter % 1000 == 0 || iter == max_iter-1){
			error = compute_error(u,f,Au);
			std::cout << "Iteration: "<< iter << " Error: " << error << std::endl;
		}		
		iter += 1;
	}
	return error;
}


int main(int argc, char** argv) {

	std::cout << "GAUSS_SEIDEL" << std::endl;
	for (long i = 0; i < 5; i++) {
		N = n * pow(2,i);
		double *u = (double*) malloc((N+2) * (N+2) * sizeof(double));
		double *f = (double*) malloc((N+2) * (N+2) * sizeof(double));
		double *temp = (double*) malloc((N+2) * (N+2) * sizeof(double));
		double *Au = (double*) malloc((N+2) * (N+2) * sizeof(double));

		for (long k = 0; k < N+2; k++) {
  			for (long l = 0; l < N+2; l++) {
  				u[k + l*(N+2)] = 0;
				f[k + l*(N+2)] = 1;
				temp[k + l*(N+2)] = 0;
				Au[k + l*(N+2)] = 1;
			}
		}
		h = 1.0/(N+1);
  		Timer t;
  		t.tic();
  		double err = gauss_seidel(u,f,temp,Au);
  		double time = t.toc();
		std::cout << "N = " << N << " time = " << time << " res_err = " << err << std::endl; 
		//std::cout << "Total time: " << time << std::endl;
		free(temp);
		free(u);
		free(f);
		free(Au);
	}

	return 0;
}
