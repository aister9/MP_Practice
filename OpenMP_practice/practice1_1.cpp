#include <omp.h>
#include <iostream>
#include <string>

#include <random>

#include "TinyLogger.h"

#define ARRAY_SIZE 1024*1024*128

void dataGen(double *data, size_t dataSize) {
	std::random_device rd;

	std::mt19937_64 gen(rd());

	std::uniform_real_distribution<double> dis(-1, 1);

	for (int i = 0; i < dataSize; i++) {
		data[i] = dis(gen);
	}
}

bool isEqual(double* A, double* B, size_t dataSize) {
	for (int i = 0; i < dataSize; i++) {
		if (std::abs(A[i] - B[i]) > 1e-8) {
			std::cout << i<< " ::" << A[i] << "!=" << B[i] << std::endl;
			return false;
		}
	}
	return true;
}

int main(int argc, char* argv[]) {
	int thread_count = 8;

	if (argc >= 2) {
		thread_count = std::atoi(argv[1]);
	}

#pragma omp parallel num_threads(thread_count)
	{
		printf("[Thread %d/%d] Hello OpenMP!\n", omp_get_thread_num(), omp_get_num_threads());
	}

	TinyLogger::Logger log;
	
	double* A = new double[ARRAY_SIZE]; dataGen(A, ARRAY_SIZE);
	double* B = new double[ARRAY_SIZE]; dataGen(B, ARRAY_SIZE);

	double* C_Serial= new double[ARRAY_SIZE];

	TinyLogger::Timer::TimePoint t_begin = TinyLogger::Timer::now();
	for (int i = 0; i < ARRAY_SIZE; i++) {
			C_Serial[i] = A[i] + B[i];
	}
	TinyLogger::Timer::TimePoint t_end = TinyLogger::Timer::now();;
	auto msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["Serial"].push_back(msec);

	double* C_Parallel = new double[ARRAY_SIZE];

	t_begin = TinyLogger::Timer::now();
	int interval = ARRAY_SIZE / thread_count;
#pragma omp parallel num_threads(thread_count)
	{
		int threadID = omp_get_thread_num();

		int start = threadID * interval;
		int end = (threadID + 1) * interval;
		if (threadID == thread_count - 1) end = ARRAY_SIZE;

		for (int i = start; i < end; i++) {
				C_Parallel[i] = A[i] + B[i];
		}
	}
	t_end = TinyLogger::Timer::now();;
	msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["Parallel"].push_back(msec);


#pragma omp parallel num_threads(thread_count)
	{
#pragma omp for
		for (int i = 0; i < ARRAY_SIZE; i++)
		{
			C_Parallel[i] = A[i] + B[i];
		}
	}


	if (isEqual(C_Serial, C_Parallel, ARRAY_SIZE)) {
		std::cout << "Good" << std::endl;
	}
	else {
		std::cout << "Is Differnt!" << std::endl;
	}

	log.print();

	delete A; delete B; delete C_Parallel; delete C_Serial;

	return 0;
}
