#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>

/**
 * CUDA kernel function: Computes AXPY (A times X Plus Y).
 *
 * Use non-consecutive memory access for each thread by incrementing with n_threads.
 */
__global__ void k_axpy_increment_threads(int n, float a, float *x, float *y) {
  int index_begin     = blockDim.x * blockIdx.x + threadIdx.x;
  int index_end       = n;
  int index_increment = blockDim.x * gridDim.x;
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

  float *d_x=NULL, *d_y=NULL;
  cudaEvent_t event_begin, event_end;
  float elapsed_time_ms=0;
  int n_threads=-1;
  int nb, nt;

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

  /* set number of threads and blocks */
  nt = 256;
  nb = 1024;
  n_threads = nb*nt;
  /* set reference solution */
  y_true = 2.4 + 11*1.0*1.3;
  /* create CUDA events for timing */
  cudaEventCreate(&event_begin);
  cudaEventCreate(&event_end);
  /* allocate arrays in device memory */
  cudaMalloc(&d_x, length*sizeof(float));
  cudaMalloc(&d_y, length*sizeof(float));

  printf("========================================\n");
  fflush(stdout); // flush buffered print outputs

  /* transfer data from host to device memory */
  cudaMemcpy(d_x, h_x, length*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, length*sizeof(float), cudaMemcpyHostToDevice);
  /* warm-up run */
  k_axpy_increment_threads<<< nb, nt >>>(length, 1.0, d_x, d_y);
  /* begin timing */
  cudaEventRecord(event_begin, 0);
  /* timing run */
  for (int k=1; k<11; k++) {
    k_axpy_increment_threads<<< nb, nt >>>(length, 1.0, d_x, d_y);
  }
  /* wait for GPU threads to complete */
  cudaDeviceSynchronize();
  /* end timing */
  cudaEventRecord(event_end, 0);
  cudaEventSynchronize(event_end);
  cudaEventElapsedTime(&elapsed_time_ms, event_begin, event_end);
  /* transfer data from device to host memory */
  cudaMemcpy(h_y, d_y, length*sizeof(float), cudaMemcpyDeviceToHost);
  /* print timing result */
  printf("k_axpy_increment_threads:\t"
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

  /* deallocate arrays in device memory */
  cudaFree(d_x);
  cudaFree(d_y);

  /* deallocate arrays in host memory */
  free(h_x);
  free(h_y);

  return 0;
}