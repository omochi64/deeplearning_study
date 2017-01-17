#pragma once

#define SIMPLE_TIMER_ENABLE

#ifdef SIMPLE_TIMER_ENABLE

#include <time.h>

class SimpleTimer
{
public:
	SimpleTimer()
		: begin_(clock())
	{
	}

	double past_seconds() const
	{
		return (double)(clock() - begin_) / CLOCKS_PER_SEC;
	}

private:
	clock_t begin_;
};

#else

class SimpleTimer
{
public:
	SimpleTimer() {}

	double past_seconds() const {
		return 0;
	}
};

#endif


