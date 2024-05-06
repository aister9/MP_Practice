#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef M_SIZE
#define M_SIZE 1024
#endif

#ifndef K_SIZE
#define K_SIZE 2048
#endif

#ifndef N_SIZE
#define N_SIZE 1024
#endif


enum MAT_MUL_MODE {
	BASE,
	WITH_SHARED,
	OPTIMIZE,
};

void matMUL_CUDA(float* a, float* b, float* c, size_t m_size, size_t k_size, size_t n_size, MAT_MUL_MODE mode);
