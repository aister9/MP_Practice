#pragma once
#include <random>
#include <iostream>
#include <omp.h>

class Histogram{
	float* data;
	size_t data_size;

	int* bins_serial;
	int* bins_parallel_ver1;
	int* bins_parallel_ver2;
	int* bins_parallel_ver3;
	size_t bin_size;

	void dataGen() {
		std::random_device rd;

		std::mt19937_64 gen(rd());

		std::uniform_real_distribution<double> dis(0, 10);

		for (int i = 0; i < data_size; i++) {
			data[i] = dis(gen);
		}
	}

	// [rl, rr) covered
	inline const unsigned int getBinIDX(const float rl, const float rr, const float value) {
		/*
			rl + (rr-rl)/bin_size * i < value <= rl+(rr-rl)/bin_size * (i+1)
		*/
		return (int)((value - rl) * bin_size / (rr - rl));
	}

public:

	Histogram(size_t _data_size, size_t _bin_size) : data_size(_data_size), bin_size(_bin_size) {
		data = new float[data_size]();
		bins_serial = new int[bin_size]();
		bins_parallel_ver1 = new int[bin_size]();
		bins_parallel_ver2 = new int[bin_size]();
		bins_parallel_ver3 = new int[bin_size]();

		//data generation
		dataGen();

		//initalize to zero
		int* results[4] = { bins_serial, bins_parallel_ver1, bins_parallel_ver2, bins_parallel_ver3 };
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < bin_size; j++)
				results[i][j] = 0;
		}

	}

	bool isValid() {
		for (int i = 0; i < bin_size; i++) {
			bool check = (bins_serial[i] == bins_parallel_ver1[i]) 
				&& (bins_serial[i] == bins_parallel_ver2[i]) 
				&& (bins_serial[i] == bins_parallel_ver3[i]);
			if (!check) return false;
		}
		return true;
	}

	void calcHist_Serial() {
		for (int i = 0; i < data_size; i++) {
			unsigned int idx = getBinIDX(0, 10, data[i]);
			bins_serial[idx]++;
		}
	}

	void calcHist_Parallel_ver1(int thread_count) {
#pragma omp parallel for num_threads(thread_count)
		for (int i = 0; i < data_size; i++) {
			unsigned int idx = getBinIDX(0, 10, data[i]);
#pragma omp atomic
			bins_parallel_ver1[idx]++;
		}
	}

	void calcHist_Parallel_ver2(int thread_count) {
		std::vector<std::vector<int>> localBins(thread_count);
		for (int i = 0; i < thread_count; i++) {
			localBins[i] = std::vector<int>(bin_size, 0);
		}

#pragma omp parallel num_threads(thread_count)
		{
			int tID = omp_get_thread_num();

		#pragma omp for
			for (int i = 0; i < data_size; i++) {
				unsigned int idx = getBinIDX(0, 10, data[i]);
				localBins[tID][idx]++;
			}

			for (int i = 0; i < bin_size; i++) {
			#pragma omp atomic
				bins_parallel_ver2[i] += localBins[tID][i];
			}
		}
	}

	void calcHist_Parallel_ver3(int thread_count) {
		std::vector<std::vector<int>> localBins(thread_count);
		for (int i = 0; i < thread_count; i++) {
			localBins[i] = std::vector<int>(bin_size, 0);
		}

#pragma omp parallel num_threads(thread_count)
		{
			int tID = omp_get_thread_num();

#pragma omp for
			for (int i = 0; i < data_size; i++) {
				unsigned int idx = getBinIDX(0, 10, data[i]);
				localBins[tID][idx]++;
			}

			for (int offset = 1; offset < thread_count; offset *= 2) {
				if (tID % (2 * offset) == 0)
				{
					for (int i = 0; i < bin_size; i+=2) {
						localBins[tID][i] += localBins[tID + offset][i];
					}
				}
				if (tID % (2 * offset) == (2 * offset-1))
				{
					for (int i = 1; i < bin_size; i+=2) {
						localBins[tID][i] += localBins[tID - offset][i];
					}
				}

#pragma omp barrier
			}
		}

		int chk[2] = { 0, thread_count - 1 };
		for (int i = 0; i < bin_size; i++) {
			bins_parallel_ver3[i] = localBins[chk[i%2]][i];
		}
	}


	void print() {
		int* results[4] = { bins_serial, bins_parallel_ver1, bins_parallel_ver2, bins_parallel_ver3 };
		std::string name_result[4] = { "Histogram_Serial", "Histogram_Parallel_ver1", "Histogram_Parallel_ver2", "Histogram_Parallel_ver2"};
		for (int i = 0; i < 4; i++) {
			std::cout << name_result[i] << std::endl;
			for (int j = 0; j < bin_size; j++) {
				std::cout << results[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}
};
