#ifndef __APPROXFLOW_COMMON__
#define __APPROXFLOW_COMMON__

#define __APPROXFLOW_DEBUGGING__ 1

#define __APPROXFLOW_SHOW_ERROR__ 1

#define __APPROXFLOW_BLOCK_SIZE_TRANSPOSE__ 64

#define __APPROXFLOW_BLOCK_SIZE_MULTIPLY__ 32

#define __APPROXFLOW_EPSILON__ 1e-8

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <array>
#include <vector>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <ctime>

// #define NDEBUG // Uncomment it to disable assert
#include <assert.h>

// #include <omp.h>

// Linux/Unix only

#include <unistd.h>

namespace ApproxFlow
{
    template<typename Type>
    void print(const std::vector<Type> &vec)
    {
        std::cout << "["; 
        for(size_t idx = 0; idx < vec.size(); idx++)
        {
            std::cout << vec[idx] << ", "; 
        }
        std::cout << "]" << std::endl; 
    }
    
    template<>
    void print<unsigned char>(const std::vector<unsigned char> &vec)
    {
        std::cout << "["; 
        for(size_t idx = 0; idx < vec.size(); idx++)
        {
            std::cout << static_cast<unsigned>(vec[idx]) << ", "; 
        }
        std::cout << "]" << std::endl; 
    }
}

#endif // __STOCHFLOW_COMMON__
