#include <iostream>
#include <string>

#include "TinyLogger.h"
#include "Histogram.h"

#define BIN_SIZE 10

int main(int argc, char* argv[]) {
	int thread_count = 8;

	if (argc >= 2) {
		thread_count = std::atoi(argv[1]);
	}

	TinyLogger::Logger log;

	Histogram histogram(1024*1024*1024, 20);

	std::cout << "Data gen complete !" << std::endl;

	//size_t* bins = new size_t[BIN_SIZE];

	TinyLogger::Timer::TimePoint t_begin = TinyLogger::Timer::now();
	histogram.calcHist_Serial();
	TinyLogger::Timer::TimePoint t_end = TinyLogger::Timer::now();;
	auto msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["Histgoram_Serial"].push_back(msec);

	t_begin = TinyLogger::Timer::now();
	histogram.calcHist_Parallel_ver1(thread_count); // using omp atomic
	t_end = TinyLogger::Timer::now();;
	msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["Histgoram_Parallel_ver1(atomic)"].push_back(msec);

	t_begin = TinyLogger::Timer::now();
	histogram.calcHist_Parallel_ver2(thread_count); // using omp atomic
	t_end = TinyLogger::Timer::now();;
	msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["Histgoram_Parallel_ver2(LocalBin+atomic)"].push_back(msec);


	t_begin = TinyLogger::Timer::now();
	histogram.calcHist_Parallel_ver3(thread_count); // using omp atomic
	t_end = TinyLogger::Timer::now();;
	msec = TinyLogger::Timer::countMicroseconds(t_begin, t_end) / 1000.f;

	log.data["Histgoram_Parallel_ver3(LocalBin+Reduction)"].push_back(msec);

	histogram.print();

	std::cout << "Validation result : " << ((histogram.isValid()) ? "Good" : "False") << std::endl;

	log.print();

	//delete A; delete b; delete C_Parallel; delete C_Serial;

	return 0;
}
