#include <iostream>
#include <vector>
#include <random>

#include <chrono>

#include "matAdd.h"

#define nCOL 8192
#define nROW 8192

void dataGen(int *data, int size) {
	std::random_device rd;

	std::mt19937_64 gen(rd());

	std::uniform_real_distribution<double> dis(0, 10);

	for (int i = 0; i < nCOL*nROW; i++) {
		data[i] = dis(gen);
	}
}

void matAdd(int* a, int* b, int* c) {
	for (int i = 0; i < nROW; i++) {
		for (int j = 0; j < nCOL; j++) {
			c[i* nCOL +j] = a[i * nCOL + j] + b[i * nCOL + j];
		}
	}
}

bool verify(int* gt, int* compare) {
	for (int i = 0; i < nROW * nCOL; i++) {
		if (gt[i] != compare[i]) {
			std::cout << "Is different ! " << gt[i] << " - " << compare[i] << std::endl;
			return false;
		}
	}
	std::cout << "All result correct !" << std::endl;
	return true;
}

int main() {
	int *a = new int[nCOL*nROW];
	int *b = new int[nCOL*nROW];
	int* c = new int[nCOL * nROW];
	int* gt = new int[nCOL * nROW];

	dataGen(a, nCOL * nROW);
	dataGen(b, nCOL * nROW);

	matADD_CUDA(a, b, c, nCOL, nROW, MAT_ADD_MODE::G2D_B2D); //warm up
	
	auto serialBegin = std::chrono::high_resolution_clock::now();
	matAdd(a, b, gt);
	auto serialEnd = std::chrono::high_resolution_clock::now();
	auto msTime = std::chrono::duration_cast<std::chrono::microseconds>(serialEnd - serialBegin).count() / 1000.;

	std::cout << "Serial computation time : " << msTime << " ms " << std::endl;

	for (int i = 0; i < 3; i++) {
		matADD_CUDA(a, b, c, nCOL, nROW, MAT_ADD_MODE(i));
		verify(gt, c);
	}

	return 0;
}
