#include "core/Timer.h"

namespace idl
{
    void Timer::start()
    {
        _startTime = std::chrono::high_resolution_clock::now();
    }

    void Timer::stop()
    {
        _endTime = std::chrono::high_resolution_clock::now();
    }

    double Timer::elapsedUs() const
    {
        return std::chrono::duration<double, std::micro>(_endTime - _startTime).count();
    }

    double Timer::elapsedMs() const
    {
        return std::chrono::duration<double, std::milli>(_endTime - _startTime).count();
    }
} // namespace
