#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>

/**
 * Computes AXPY (A times X Plus Y).
 *
 * Use non-consecutive memory access for each thread by incrementing with n_threads.
 */
void axpy_increment_threads(int n, float a, float *x, float *y) {
  int index_begin     = omp_get_thread_num();
  int index_end       = n;
  int index_increment = omp_get_num_threads();
  int i;

  for (i=index_begin; i<index_end; i+=index_increment) {
    y[i] = a*x[i] + y[i];
  }
}

/**
 * Main function.
 */
int main(int argc, char **argv) {
  int length;
  float *h_x=NULL, *h_y=NULL;
  float y_true = NAN;

  int n_threads=-1;
  double elapsed_time_ms=0;

  /* read problem size/length from command line */
  if (2 <= argc) {
    length = strtol(argv[1], NULL, 10);
  }
  printf("Problem size: length=%g\n", (double) length);
  fflush(stdout); // flush buffered print output

  /* allocate arrays in host memory */
  h_x = (float*) malloc(length*sizeof(float));
  h_y = (float*) malloc(length*sizeof(float));

  /* initialize x and y arrays */
  for (int i=0; i<length; i++) {
    h_x[i] = 1.3;
    h_y[i] = 2.4;
  }

  /* set reference solution */
  y_true = 2.4 + 11*1.0*1.3;

  printf("========================================\n");
  fflush(stdout); // flush buffered print outputs

  /* warm-up run */
#pragma omp parallel
  axpy_increment_threads(length, 1.0, h_x, h_y);
  /* begin timing */
  elapsed_time_ms -= 1000.0*omp_get_wtime();
  /* timing run */
  for (int k=1; k<11; k++) {
#pragma omp parallel
    axpy_increment_threads(length, 1.0, h_x, h_y);
  }
  /* end timing */
  elapsed_time_ms += 1000.0*omp_get_wtime();
  /* print timing result */
#pragma omp parallel
  n_threads = omp_get_num_threads();
  printf("axpy_increment_threads:\t"
         "Number of threads %d,\tWall-clock time [ms] ~ %g\n",
         n_threads, elapsed_time_ms);

  /* check errors */
  float max_error = 0.0;
  for (int i=0; i<length; i++) {
    max_error = fmax(max_error, fabs(h_y[i] - y_true));
  }
  printf("Max error = %e\n", max_error);

  printf("========================================\n");
  fflush(stdout); // flush buffered print outputs


  /* deallocate arrays in host memory */
  free(h_x);
  free(h_y);

  return 0;
}