#include <omp.h>
#include <iostream>
#include <string>

#include <random>

#include "TinyLogger.h"

#define ROW 10000
#define COL 10000

#define CHK 8

inline const double xsquare(const double x) {
	return x * x;
}

inline const double trapezoidal(const double bottom, const double up, const double height) {
	return (bottom + up) * height * 0.5;
}

void dataGen(float* data, size_t dataSize) {
	std::random_device rd;

	std::mt19937_64 gen(rd());

	std::uniform_real_distribution<double> dis(-1, 1);

	for (int i = 0; i < dataSize; i++) {
		data[i] = dis(gen);
	}
}

bool isEqual(float* A, float* B, size_t dataSize) {
	for (int i = 0; i < dataSize; i++) {
		if (std::fabsf(A[i] - B[i]) > 1e-5f) {
			std::cout << i << " ::" << A[i] << "!=" << B[i] << std::endl;
			return false;
		}
	}
	return true;
}

void MatVecMul(float** A, float* b, float* c, int thread_count, bool isParallel) {
#pragma omp parallel for num_threads(thread_count) if(isParallel)
	for (int yy = 0; yy < ROW; yy++) {
		c[yy] = 0.0f;
		for (int xx = 0; xx < COL; xx++) {
			c[yy] += A[yy][xx] * b[xx];
		}
	}
}

void MatVecMulVer2(float** A, float* b, float* c, int thread_count) {
	float* threadLocalSum = new float[thread_count];
	for (int yy = 0; yy < ROW; yy++) {
		c[yy] = 0.0f;
		#pragma omp parallel num_threads(thread_count)
		{
			int tID = omp_get_thread_num();
			threadLocalSum[tID] = 0;
			float localSum = 0.0f;
			#pragma omp for
			for (int xx = 0; xx < COL; xx++) {
				localSum += A[yy][xx] * b[xx];
			}
			
			threadLocalSum[tID] = localSum;
		}
		for (int idx = 0; idx < thread_count; idx++)
			c[yy] += threadLocalSum[idx];
	}
	delete threadLocalSum;
}

double TrapezoidalRule(int a, int b, int n, int thread_count, bool isParallel) {
	double globalSum = 0.0f;
	double*localSum = new double[thread_count];

	double delta = (double)(b - a) / (double)n;

#pragma omp parallel num_threads(thread_count) if(isParallel)
	if (omp_in_parallel) {
		int tid = omp_get_thread_num();
		localSum[tid] = 0.0;

		//optimize 1.
		double threadLocalSum = 0.0;

#pragma omp for
		for (int i = 0; i < n; i++) {
			//localSum[tid] += trapezoidal(xsquare(a + i * delta), xsquare(a + (i + 1) * delta), delta);
			threadLocalSum += trapezoidal(xsquare(a + i * delta), xsquare(a + (i + 1) * delta), delta);
		}
		
		localSum[tid] = threadLocalSum; //sync

#pragma omp barrier //*important
#pragma omp single
		{
			//global sum
			for (int i = 0; i < thread_count; i++) {
				globalSum += localSum[i];
			}
		}
	}
	else {
		for (int i = 0; i < n; i++) {
			globalSum += trapezoidal(xsquare(a + i * delta), xsquare(a + (i + 1) * delta), delta);
		}
	}

	
	return globalSum;
}

double TrapezoidalRule_CacheOptimize(int a, int b, int n, int thread_count) {
	double globalSum = 0.0f;
	double* localSum = new double[thread_count*CHK];

	double delta = (double)(b - a) / (double)n;

#pragma omp parallel num_threads(thread_count) 
	{
		int tid = omp_get_thread_num() * CHK;
		localSum[tid] = 0.0;
#pragma omp for
		for (int i = 0; i < n; i++) {
			//localSum[tid] += trapezoidal(xsquare(a + i * delta), xsquare(a + (i + 1) * delta), delta);
			localSum[tid] += trapezoidal(xsquare(a + i * delta), xsquare(a + (i + 1) * delta), delta);
		}
	}
	for (int i = 0; i < thread_count; i++) {
		globalSum += localSum[i * CHK];
	}

	return globalSum;
}

double TrapezoidalRule_Reduction(int a, int b, int n, int thread_count) {
	double globalSum = 0.0f;

	double delta = (double)(b - a) / (double)n;

#pragma omp parallel for num_threads(thread_count) reduction(+:globalSum)
		for (int i = 0; i < n; i++) {
			globalSum += trapezoidal(xsquare(a + i * delta), xsquare(a + (i + 1) * delta), delta);
		}

	return globalSum;
}

int main(int argc, char* argv[]) {
	int thread_count = 16;

	if (argc >= 2) {
		thread_count = std::atoi(argv[1]);
	}

	TinyLogger::Logger log;

	float** A = new float*[ROW];
	for (int yy = 0; yy < ROW; yy++) {
		A[yy] = new float[COL];
		dataGen(A[yy], COL);
	}
	float* b = new float[COL]; dataGen(b, COL);

	float* C_Serial = new float[COL];

	TinyLogger::Timer::TimePoint t_begin = TinyLogger::Timer::now();
	MatVecMul(A, b, C_Serial, thread_count, false);
	TinyLogger::Timer::TimePoint t_end = TinyLogger::Timer::now();;
	auto msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["MatVec_Serial"].push_back(msec);

	float* C_Parallel = new float[COL];
	float* C_Parallel2 = new float[COL];

	t_begin = TinyLogger::Timer::now();
	MatVecMul(A, b, C_Parallel, thread_count, true);
	t_end = TinyLogger::Timer::now();;
	msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["MatVec_Parallel"].push_back(msec);

	if (isEqual(C_Serial, C_Parallel, COL)) {
		std::cout << "Good" << std::endl;
	}
	else {
		std::cout << "Is Differnt!" << std::endl;
	}

	{
		double res_serial, res_parallel, res_parallel_ver2, res_parallel_ver3;
		int a = 6; int b = 10; int n = 1024 * 1024 * 1024;

		TinyLogger::Timer::TimePoint t_begin = TinyLogger::Timer::now();
		res_serial = TrapezoidalRule(a, b, n, thread_count, false);
		TinyLogger::Timer::TimePoint t_end = TinyLogger::Timer::now();;
		auto msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

		log.data["Trapezoidal_Serial"].push_back(msec);

		t_begin = TinyLogger::Timer::now();
		res_parallel = TrapezoidalRule(a, b, n, thread_count, true);
		t_end = TinyLogger::Timer::now();;
		msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

		log.data["Trapezoidal_Parallel"].push_back(msec);

		t_begin = TinyLogger::Timer::now();
		res_parallel_ver2 = TrapezoidalRule_CacheOptimize(a, b, n, thread_count);
		t_end = TinyLogger::Timer::now();;
		msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

		log.data["Trapezoidal_Parallel_CacheOptimize"].push_back(msec);

		t_begin = TinyLogger::Timer::now();
		res_parallel_ver3 = TrapezoidalRule_CacheOptimize(a, b, n, thread_count);
		t_end = TinyLogger::Timer::now();;
		msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

		log.data["Trapezoidal_Parallel_reduction"].push_back(msec);

		std::cout << "[Trapezoidal] Serial result : " << res_serial << std::endl;
		std::cout << "[Trapezoidal] Parallel result : " << res_parallel << std::endl;
		std::cout << "[Trapezoidal] Parallel_CacheOptimize result : " << res_parallel_ver2 << std::endl;
		std::cout << "[Trapezoidal] Parallel_reduction result : " << res_parallel_ver3 << std::endl;
	}

	log.print();

	//delete A; delete b; delete C_Parallel; delete C_Serial;

	return 0;
}
