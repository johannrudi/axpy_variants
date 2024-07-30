#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cublas_v2.h>


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
  const float a=1.0;
  cublasHandle_t cublasH=NULL;

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
  /* create CUDA events for timing */
  cudaEventCreate(&event_begin);
  cudaEventCreate(&event_end);
  /* create cuBLAS handle */
  cublasCreate(&cublasH);
  /* allocate arrays in device memory */
  cudaMalloc(&d_x, length*sizeof(float));
  cudaMalloc(&d_y, length*sizeof(float));

  printf("========================================\n");
  fflush(stdout); // flush buffered print outputs

  /* transfer data from host to device memory */
  cudaMemcpy(d_x, h_x, length*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, length*sizeof(float), cudaMemcpyHostToDevice);
  /* warm-up run */
  cublasSaxpy(cublasH, length, &a, d_x, 1, d_y, 1);
  /* begin timing */
  cudaEventRecord(event_begin, 0);
  /* timing run */
  for (int k=1; k<11; k++) {
    cublasSaxpy(cublasH, length, &a, d_x, 1, d_y, 1);
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
  printf("cublasSaxpy:\t"
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