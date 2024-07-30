# set compilers
CC=gcc
CU=nvcc
# set remove command
RM=rm

# set compiler flags
CFLAGS=-I. -O3
# set library directory flags
LDFLAGS=
# set library flags
LIBS=-lm

# set OpenMP and CUDA flags
OMP_CFLAGS=-fopenmp -DMKL_ILP64 -m64 -I"${MKLROOT}/include"
OMP_BLAS_LIBS=-m64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
CUDA_CFLAGS=#-arch=sm_XX
CUDA_BLAS_LIBS=-lcublas

# set the program name
PROGRAM_OMP1=openmp_increment_1
PROGRAM_OMP2=openmp_increment_threads
PROGRAM_OMP3=openmp_blas
PROGRAM_CUDA1=cuda_increment_1
PROGRAM_CUDA2=cuda_increment_threads
PROGRAM_CUDA3=cuda_single_iter
PROGRAM_CUDA4=cuda_blas

# list object files
OBJS_OMP1=openmp_increment_1.o
OBJS_OMP2=openmp_increment_threads.o
OBJS_OMP3=openmp_blas.o
OBJS_CUDA1=cuda_increment_1.o
OBJS_CUDA2=cuda_increment_threads.o
OBJS_CUDA3=cuda_single_iter.o
OBJS_CUDA4=cuda_blas.o

# list header files
DEPS=

.PHONY: all
all: $(PROGRAM_OMP1) $(PROGRAM_OMP2) $(PROGRAM_OMP3) $(PROGRAM_CUDA1) $(PROGRAM_CUDA2) $(PROGRAM_CUDA3) $(PROGRAM_CUDA4)

$(PROGRAM_OMP1): $(OBJS_OMP1)
	$(CC) $(LDFLAGS) -o $@ $^ $(OMP_CFLAGS) $(LIBS)

$(PROGRAM_OMP2): $(OBJS_OMP2)
	$(CC) $(LDFLAGS) -o $@ $^ $(OMP_CFLAGS) $(LIBS)

$(PROGRAM_OMP3): $(OBJS_OMP3)
	$(CC) $(LDFLAGS) -o $@ $^ $(OMP_CFLAGS) $(LIBS) $(OMP_BLAS_LIBS)

$(PROGRAM_CUDA1): $(OBJS_CUDA1)
	$(CU) $(LDFLAGS) -o $@ $^ $(CUDA_CFLAGS) $(LIBS)

$(PROGRAM_CUDA2): $(OBJS_CUDA2)
	$(CU) $(LDFLAGS) -o $@ $^ $(CUDA_CFLAGS) $(LIBS)

$(PROGRAM_CUDA3): $(OBJS_CUDA3)
	$(CU) $(LDFLAGS) -o $@ $^ $(CUDA_CFLAGS) $(LIBS)

$(PROGRAM_CUDA4): $(OBJS_CUDA4)
	$(CU) $(LDFLAGS) -o $@ $^ $(CUDA_CFLAGS) $(LIBS) $(CUDA_BLAS_LIBS)

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(OMP_CFLAGS) -c -o $@ $<

%.o: %.cu $(DEPS)
	$(CU) $(CFLAGS) $(CUDA_CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	$(RM) -f $(OBJS_OMP1) $(PROGRAM_OMP1)
	$(RM) -f $(OBJS_OMP2) $(PROGRAM_OMP2)
	$(RM) -f $(OBJS_OMP3) $(PROGRAM_OMP3)
	$(RM) -f $(OBJS_CUDA1) $(PROGRAM_CUDA1)
	$(RM) -f $(OBJS_CUDA2) $(PROGRAM_CUDA2)
	$(RM) -f $(OBJS_CUDA3) $(PROGRAM_CUDA3)
	$(RM) -f $(OBJS_CUDA4) $(PROGRAM_CUDA4)
