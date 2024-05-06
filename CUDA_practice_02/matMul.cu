#include "matMul.h"
#include "TinyLogger.h"

__global__ void kernel_matmul_base(float* a, float* b, float* c, size_t m_size, size_t k_size, size_t n_size) {
	const unsigned int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idxx >= m_size || idxy >= n_size)
		return;

	const unsigned int tIDX = idxx * n_size + idxy;
	
	c[tIDX] = 0;
	for (int k = 0; k < k_size; k++) {
		c[tIDX] += a[idxx * k_size + k] * b[k * n_size + idxy];
	}
}

__global__ void kernel_matmul_shared(float* a, float* b, float* c, size_t m_size, size_t k_size, size_t n_size) {
	const unsigned int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idxx >= m_size || idxy >= n_size)
		return;

	__shared__ float sA[16][16];
	__shared__ float sB[16][16];

	const unsigned int tIDX = idxx * n_size + idxy;

	float local_c = 0.0;

	for (int k = 0; k < k_size; k+=16) {
		const unsigned int localIDXx = (k + threadIdx.x);
		const unsigned int localIDXy = (k + threadIdx.y);

		if (localIDXy >= k_size) sA[threadIdx.x][threadIdx.y] = 0;
		else sA[threadIdx.x][threadIdx.y] = a[idxx * k_size + localIDXy];

		if (localIDXx >= k_size) sB[threadIdx.x][threadIdx.y] = 0;
		else sB[threadIdx.x][threadIdx.y] = b[localIDXx * n_size + idxy];

		__syncthreads();

		for (int b = 0; b < 16; b++) {
			local_c += sA[threadIdx.x][b] * sB[b][threadIdx.y];
		}

		__syncthreads();
	}
	c[tIDX] = local_c;
}

__global__ void kernel_matmul_shared_optimized(float* a, float* b, float* c, size_t m_size, size_t k_size, size_t n_size) {
	const unsigned int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idxx >= m_size || idxy >= n_size)
		return;

	__shared__ float sA[16][16];
	__shared__ float sB[16][16];

	const unsigned int tIDX = idxy * n_size + idxx;

	float local_c = 0.0;

	for (int k = 0; k < k_size; k += 16) {
		const unsigned int localIDXx = (k + threadIdx.x);
		const unsigned int localIDXy = (k + threadIdx.y);

		if (localIDXx >= k_size) sA[threadIdx.y][threadIdx.x] = 0;
		else sA[threadIdx.y][threadIdx.x] = a[idxy * k_size + localIDXx];

		if (localIDXy >= k_size) sB[threadIdx.y][threadIdx.x] = 0;
		else sB[threadIdx.y][threadIdx.x] = b[localIDXy * n_size + idxx];

		__syncthreads();

		for (int b = 0; b < 16; b++) {
			local_c += sA[threadIdx.y][b] * sB[b][threadIdx.x];
		}

		__syncthreads();
	}
	c[tIDX] = local_c;
}


void matMUL_CUDA(float* a, float* b, float* c, size_t m_size, size_t k_size, size_t n_size, MAT_MUL_MODE mode) {
	float* d_a, * d_b, * d_c;

	TinyLogger::Logger log;

	std::vector<std::string> mode_s = { "BASE","SHARED","OPTIMIZE" };

	std::vector<dim3> blocks = { dim3(16,16), dim3(16,16), dim3(16,16) };
	std::vector<dim3> grids = { dim3(ceil((float)m_size / blocks[0].x), ceil((float)n_size / blocks[0].y)),
		dim3(ceil((float)m_size / blocks[1].x), ceil((float)n_size / blocks[1].y)),
		dim3(ceil((float)m_size / blocks[2].x), ceil((float)n_size / blocks[2].y))
	};

	void (*p_kernel[3])(float* , float* , float* , size_t, size_t, size_t) = { &kernel_matmul_base,  &kernel_matmul_shared , &kernel_matmul_shared_optimized };

	std::cout << "*----------------------CUDA COMPUTE----------------------*" << std::endl;
	std::cout << "*---------------------MODE : " << mode_s[mode] << "---------------------*" << std::endl;

	TinyLogger::Timer::TimePoint tBegin = TinyLogger::Timer::now();
	cudaMalloc(&d_a, sizeof(float) * m_size * k_size);
	cudaMalloc(&d_b, sizeof(float) * k_size * n_size);
	cudaMalloc(&d_c, sizeof(float) * m_size * n_size);
	cudaMemcpy(d_a, a, sizeof(float) * m_size * k_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * k_size * n_size, cudaMemcpyHostToDevice);
	TinyLogger::Timer::TimePoint tEnd = TinyLogger::Timer::now();
	auto msTime = TinyLogger::Timer::countMicroseconds(tBegin, tEnd) / 1000.;
	log.data["0_Memcpy(HtoD)"].push_back(msTime);

	tBegin = TinyLogger::Timer::now();
	p_kernel[mode] << <grids[mode], blocks[mode] >> > (d_a, d_b, d_c, m_size, k_size, n_size);
	cudaDeviceSynchronize();
	tEnd = TinyLogger::Timer::now();
	msTime = TinyLogger::Timer::countMicroseconds(tBegin, tEnd) / 1000.;
	log.data["1_Kernel_execution_time"].push_back(msTime);

	cudaError_t c_error = cudaGetLastError();
	if (cudaSuccess != c_error)
	{
		std::cout << cudaGetErrorString(c_error) << std::endl;
	}

	tBegin = TinyLogger::Timer::now();
	cudaMemcpy(c, d_c, sizeof(float) * m_size * n_size, cudaMemcpyDeviceToHost);
	tEnd = TinyLogger::Timer::now();
	msTime = TinyLogger::Timer::countMicroseconds(tBegin, tEnd) / 1000.;
	log.data["2_Memcpy(DtoH)"].push_back(msTime);

	std::string gridSize = "(" + std::to_string(grids[mode].x) + "," + std::to_string(grids[mode].y) + "),(" + std::to_string(blocks[mode].x) + "," + std::to_string(blocks[mode].y) + ")";
	log.data["3_Grid_size"].push_back(gridSize);

	log.print();
	std::cout << "*--------------------------------------------------------*" << std::endl;

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
