#!/bin/bash

LENGTH=100100100

printf "\nRun OpenMP version 1\n\n"
export OMP_NUM_THREADS=1   && ./openmp_increment_1 $LENGTH
export OMP_NUM_THREADS=2   && ./openmp_increment_1 $LENGTH
export OMP_NUM_THREADS=4   && ./openmp_increment_1 $LENGTH
export OMP_NUM_THREADS=8   && ./openmp_increment_1 $LENGTH
export OMP_NUM_THREADS=16  && ./openmp_increment_1 $LENGTH
export OMP_NUM_THREADS=32  && ./openmp_increment_1 $LENGTH

printf "\nRun OpenMP version 2\n\n"
export OMP_NUM_THREADS=1   && ./openmp_increment_threads $LENGTH
export OMP_NUM_THREADS=2   && ./openmp_increment_threads $LENGTH
export OMP_NUM_THREADS=4   && ./openmp_increment_threads $LENGTH
export OMP_NUM_THREADS=8   && ./openmp_increment_threads $LENGTH
export OMP_NUM_THREADS=16  && ./openmp_increment_threads $LENGTH
export OMP_NUM_THREADS=32  && ./openmp_increment_threads $LENGTH

printf "\nRun OpenMP version 2\n\n"
export OMP_NUM_THREADS=1   && ./openmp_blas $LENGTH
export OMP_NUM_THREADS=2   && ./openmp_blas $LENGTH
export OMP_NUM_THREADS=4   && ./openmp_blas $LENGTH
export OMP_NUM_THREADS=8   && ./openmp_blas $LENGTH
export OMP_NUM_THREADS=16  && ./openmp_blas $LENGTH
export OMP_NUM_THREADS=32  && ./openmp_blas $LENGTH

printf "\nRun CUDA version 1\n\n"
./cuda_increment_1 $LENGTH

printf "\nRun CUDA version 2\n\n"
./cuda_increment_threads $LENGTH

printf "\nRun CUDA version 3\n\n"
./cuda_single_iter $LENGTH

printf "\nRun CUDA version 4\n\n"
./cuda_blas $LENGTH
