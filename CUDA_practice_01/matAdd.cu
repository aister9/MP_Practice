#include "matAdd.h"
#include "TinyLogger.h"

__global__ void kernel_matadd_g2d_b2d(int* a, int* b, int* c, size_t nCol, size_t nRow) {
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idx = iy * nCol + ix;

	if (ix < nCol && iy < nRow)
		c[idx] = a[idx] + b[idx];
}

__global__ void kernel_matadd_g1d_b1d(int* a, int* b, int* c, size_t nCol, size_t nRow) {
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix < nCol) {
		for (int iy = 0; iy < nRow; iy++) {
			const unsigned int idx = iy * nCol + ix;
			c[idx] = a[idx] + b[idx];
		}
	}
}

__global__ void kernel_matadd_g2d_b1d(int* a, int* b, int* c, size_t nCol, size_t nRow) {
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y;
	const unsigned int idx = iy * nCol + ix;

	if (ix < nCol && iy < nRow)
		c[idx] = a[idx] + b[idx];
}

void matADD_CUDA(int* a, int* b, int* c, size_t nCol, size_t nRow, MAT_ADD_MODE mode) {
	int* d_a, * d_b, * d_c;

	TinyLogger::Logger log;

	std::vector<std::string> mode_s = { "G2D_B2D","G1D_B1D","G2D_B1D" };

	std::vector<dim3> blocks = {dim3(32,32), dim3(32), dim3(32)};
	std::vector<dim3> grids = {dim3(ceil((float)nCol/blocks[0].x), ceil((float)nRow / blocks[0].y)),
		dim3(ceil((float)nCol / blocks[1].x)),
		dim3(ceil((float)nCol / blocks[2].x), nRow)
		};

	void (*p_kernel[3])(int*, int*, int*, size_t, size_t) = { &kernel_matadd_g2d_b2d,  &kernel_matadd_g1d_b1d , &kernel_matadd_g2d_b1d };

	std::cout << "*----------------------CUDA COMPUTE----------------------*"<<std::endl;
	std::cout << "*---------------------MODE : "<< mode_s[mode] << "---------------------*" << std::endl;

	TinyLogger::Timer::TimePoint tBegin = TinyLogger::Timer::now();
	cudaMalloc(&d_a, sizeof(int) * nCol * nRow);
	cudaMalloc(&d_b, sizeof(int) * nCol * nRow);
	cudaMalloc(&d_c, sizeof(int) * nCol * nRow);
	cudaMemcpy(d_a, a, sizeof(int) * nCol * nRow, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(int) * nCol * nRow, cudaMemcpyHostToDevice);
	TinyLogger::Timer::TimePoint tEnd = TinyLogger::Timer::now();
	auto msTime = TinyLogger::Timer::countMicroseconds(tBegin, tEnd) / 1000.;
	log.data["0_Memcpy(HtoD)"].push_back(msTime);

	tBegin = TinyLogger::Timer::now();
	p_kernel[mode]<<<grids[mode], blocks[mode]>>>(d_a, d_b, d_c, nCol, nRow);
	cudaDeviceSynchronize();
	tEnd = TinyLogger::Timer::now();
	msTime = TinyLogger::Timer::countMicroseconds(tBegin, tEnd) / 1000.;
	log.data["1_Kernel_execution_time"].push_back(msTime);

	tBegin = TinyLogger::Timer::now();
	cudaMemcpy(c, d_c, sizeof(int) * nCol * nRow, cudaMemcpyDeviceToHost);
	tEnd = TinyLogger::Timer::now();
	msTime = TinyLogger::Timer::countMicroseconds(tBegin, tEnd) / 1000.;
	log.data["2_Memcpy(DtoH)"].push_back(msTime);

	std::string gridSize = "(" + std::to_string(grids[mode].x) + "," + std::to_string(grids[mode].y) + "),(" + std::to_string(blocks[mode].x) + "," + std::to_string(blocks[mode].y) + ")";
	log.data["3_Grid_size"].push_back(gridSize);

	log.print();
	std::cout << "*--------------------------------------------------------*" << std::endl;

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
