# AXPY Variants
Algorithmic variants of parallel AXPY for CPU and GPU hardware

AXPY (meaning **A** times **X** **p**lus **Y**) is a scalar-vector 
multiplication followed by a vector-vector addition:
$y_i = a x_i + y_i$, where $a\in\mathbb{R}$, $x,y\in\mathbb{R}^N$,
$N\in\mathbb{N}$, and subscript notation $x_i,y_i$ refers to entries of the
vectors $x,y$ at index $i\in\{0,\ldots,N-1\}$.

This repository contains seven variants of AXPY, all in C/C++ that differ 
in their multithreaded parallel algorithms and in the implementation of 
parallelization using either OpenMP or CUDA.
