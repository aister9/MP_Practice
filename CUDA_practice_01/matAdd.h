#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
//#include "TinyLogger.h"

enum MAT_ADD_MODE {
	G2D_B2D,
	G1D_B1D,
	G2D_B1D
};

void matADD_CUDA(int* a, int* b, int* c, size_t nCol, size_t nRow, MAT_ADD_MODE mode);