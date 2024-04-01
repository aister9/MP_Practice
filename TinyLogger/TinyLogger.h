#pragma once
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <any>
#include <variant>
#include <iostream>

#define WIN32_LEAN_AND_MEAN             // 거의 사용되지 않는 내용을 Windows 헤더에서 제외합니다.

namespace TinyLogger{
	class Timer {
	public:
		using Clock = std::chrono::high_resolution_clock;
		using TimePoint = std::chrono::time_point<Clock>;
		using Microseconds = std::chrono::microseconds;
		using Milliseconds = std::chrono::milliseconds;

		static TimePoint now() {
			return Clock::now();
		}

		static long long countMicroseconds(TimePoint start, TimePoint end) {
			return std::chrono::duration_cast<Microseconds>(end - start).count();
		}

		static float countMilliseconds(TimePoint start, TimePoint end) {
			// Here, we directly cast to milliseconds before counting, improving precision
			return std::chrono::duration_cast<Milliseconds>(end - start).count();
		}
	};

	class Logger {
	public:
		std::map<std::string, std::vector<std::variant<int, double, std::string>>> data;
		std::string filePath;

		void print();
		friend std::ostream& operator<<(std::ostream& os, const Logger& data);
	};

	void Logger::print() {
		std::cout << "------------------------------------------------------------------" << std::endl;
		for (auto& v : data) {
			std::cout << v.first << " :: ";
			std::visit([](auto&& arg) {std::cout<< arg ; }, v.second[0]);
			std::cout << " ms" << std::endl;
		}
		std::cout << "------------------------------------------------------------------" << std::endl;
	}

	std::ostream& operator<<(std::ostream& os, const Logger& data) {
		int t_size = data.data.begin()->second.size(); // it assumes that all of the map vector sizes is same 

		os << "sep=;" << std::endl;
		for (auto& v : data.data) {
			os << v.first << ";";
		}
		os << std::endl;

		for (int i = 0; i < t_size; i++) {
			for (auto& v : data.data) {
				std::visit([&os](auto&& arg) {os << arg << ";"; }, v.second[i]);
			}
			os << std::endl;
		}

		return os;
	}
}
