# AXPY Variants
Algorithmic variants of parallel AXPY for CPU and GPU hardware.

AXPY (meaning **A** times **X** **p**lus **Y**) is a scalar-vector
multiplication followed by a vector-vector addition: $y_i = a x_i + y_i$, where
$a\in\mathbb{R}$, $x,y\in\mathbb{R}^N$, $N\in\mathbb{N}$, and subscript
notation $x_i,y_i$ refers to entries of the vectors $x,y$ at index
$i\in\{0,\ldots,N-1\}$.

This repository contains seven variants of AXPY, all in C/C++, that differ in
their multithreaded parallel algorithms and in the implementation of
parallelization using either OpenMP or CUDA.
More details about the algorithms and the implementations are found in this paper:

> **CG-Kit: Code Generation Toolkit for Performant and Maintainable Variants of Source Code Applied to Flash-X Hydrodynamics Simulations**
> by Johann Rudi, Youngjun Lee, Aidan H. Chadha, Mohamed Wahib, Klaus Weide, Jared P. O'Neal, and Anshu Dubey, 2024.
> URL: https://arxiv.org/abs/2401.03378
