#include <iostream>
#include <vector>
#include <random>

#include <chrono>

#include "matMul.h"
#include <omp.h>

void dataGen(float* data, int size) {
	std::random_device rd;

	std::mt19937_64 gen(rd());

	std::uniform_real_distribution<double> dis(-1, 1);

	for (int i = 0; i < size; i++) {
		data[i] = dis(gen);
	}
}

bool verify(float* gt, float* compare, int size) {
	for (int i = 0; i < size; i++) {
		if (abs(gt[i] - compare[i]) > 1e-4f) {
			std::cout << "Is different ! " << gt[i] << " - " << compare[i] << std::endl;
			return false;
		}
	}
	std::cout << "All result correct !" << std::endl;
	return true;
}

void MatMul(float* a, float* b, float* c, size_t m_size, size_t k_size, size_t n_size) {
#pragma omp parallel for
	for (int yy = 0; yy < m_size; yy++) {
		for (int xx = 0; xx < n_size; xx++) {
			c[yy * n_size + xx] = 0;
			for (int kk = 0; kk < k_size; kk++) {
				c[yy * n_size + xx] += a[yy * k_size + kk] * b[kk * n_size + xx];
			}
		}
	}
}

int main() {
	float* a = new float[M_SIZE * K_SIZE];
	float* b = new float[K_SIZE * N_SIZE];
	float* c = new float[M_SIZE * N_SIZE];
	float* gt = new float[M_SIZE * N_SIZE];

	dataGen(a, M_SIZE * K_SIZE);
	dataGen(b, K_SIZE * N_SIZE);

	matMUL_CUDA(a, b, c, M_SIZE, K_SIZE, N_SIZE, MAT_MUL_MODE(0));

	auto serialBegin = std::chrono::high_resolution_clock::now();
	MatMul(a, b, gt, M_SIZE, K_SIZE, N_SIZE);
	auto serialEnd = std::chrono::high_resolution_clock::now();
	auto msTime = std::chrono::duration_cast<std::chrono::microseconds>(serialEnd - serialBegin).count() / 1000.;

	std::cout << "Serial computation time : " << msTime << " ms " << std::endl;

	for (int i = 0; i < 3; i++) {
		matMUL_CUDA(a, b, c, M_SIZE, K_SIZE, N_SIZE, MAT_MUL_MODE(i));

		for (int j = 0; j < 3; j++) {
			std::cout << c[j] << " - " << gt[j] << ", ";
		}
		std::cout << "..." << std::endl;
		verify(gt, c, M_SIZE*N_SIZE);
	}

	return 0;
}
