#pragma once

#include <chrono>

namespace idl
{
    class Timer
    {
        public:
            void start();

            void stop();

            double elapsedUs() const;

            double elapsedMs() const;

        private:
            std::chrono::high_resolution_clock::time_point _startTime;
            std::chrono::high_resolution_clock::time_point _endTime;
    };
} // namespace
