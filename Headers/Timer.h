#ifndef __APPROXFLOW_TIMER__
#define __APPROXFLOW_TIMER__

#include <sys/time.h>

#include "Common.h"

namespace ApproxFlow
{
class Timer
{
private: 
    struct timeval _time; 

public:
    Timer() {gettimeofday(&_time, 0); }
    void begin() {gettimeofday(&_time, 0); }
    
    double us() const
    {
        struct timeval tmp;
        gettimeofday(&tmp, 0); 
        
        return (tmp.tv_sec -_time.tv_sec) * 1e6+ tmp.tv_usec-_time.tv_usec; 
    }
    
    double ms() const
    {
        struct timeval tmp;
        gettimeofday(&tmp, 0); 
        
        return ((tmp.tv_sec -_time.tv_sec) * 1e6+ tmp.tv_usec-_time.tv_usec) / 1000.0; 
    }
    
    double s() const
    {
        struct timeval tmp;
        gettimeofday(&tmp, 0); 
        
        return ((tmp.tv_sec -_time.tv_sec) * 1e6+ tmp.tv_usec-_time.tv_usec) / 1e6; 
    }
    
}; 
}

#endif
