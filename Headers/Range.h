#ifndef __APPROXFLOW_RANGE__
#define __APPROXFLOW_RANGE__

#include <functional>

// #include <omp.h>

namespace ApproxFlow
{

template<typename Type>
class RangeType
{
private:
    Type _begin, _end; 

public:
    RangeType(); 
    RangeType(Type begin, Type end); 
    
    Type begin() const {return _begin; }
    Type end() const {return _end; }
    Type size() const {return _end - _begin; }
    Type length() const {return _end - _begin; }
    
    void foreach(std::function<void(Type)> func) const; 
    void parfor(std::function<void(Type)> func) const; 
}; 

template<typename Type>
RangeType<Type>::RangeType(): _begin(0), _end(0)
{
    ; 
}

template<typename Type>
RangeType<Type>::RangeType(Type begin, Type end): _begin(begin), _end(end)
{
#ifdef __UTILITIES_DEBUG__
    if(begin < 0 || end < 0)
    {
        std::cerr << "ERROR: RangeType::RangeType(Type begin, Type end) -> range must be positive. " << std::endl; 
        exit(22); 
    }
    if(begin > end)
    {
        std::cerr << "ERROR: RangeType::RangeType(Type begin, Type end) -> begin cannot be larger than end. " << std::endl; 
        exit(22); 
    }
#endif
    ; 
}

template<typename Type>
void RangeType<Type>::foreach(std::function<void(Type)> func) const
{
    for(Type idx = _begin; idx < _end; idx++)
    {
        func(idx); 
    }
}

template<typename Type>
void RangeType<Type>::parfor(std::function<void(Type)> func) const
{
    #pragma omp parallel for
    for(Type idx = _begin; idx < _end; idx++)
    {
        func(idx); 
    }
}

typedef RangeType<size_t> Range; 

}

#endif

