#ifndef __APPROXFLOW_FIXEDNUM__
#define __APPROXFLOW_FIXEDNUM__

#include "Common.h"

namespace ApproxFlow
{
template<size_t SIZE>
class FixedNum
{
private:
    
    size_t _val; 
    
public: 
//     static size_t Mask; 
//     static std::vector<std::vector<size_t>> LUT; 
//     static void InitLUT(); 
//     static size_t float2val(float x); 
    
    FixedNum(); 
    FixedNum(float val); 
    FixedNum(const FixedNum &val); 
    
    const FixedNum &operator = (float val); 
    const FixedNum &operator = (const FixedNum &val); 
    
    bool operator == (float val) const; 
    bool operator == (const FixedNum &val) const; 
    bool operator != (float val) const; 
    bool operator != (const FixedNum &val) const; 
    bool operator > (float val) const; 
    bool operator > (const FixedNum &val) const; 
    bool operator < (float val) const; 
    bool operator < (const FixedNum &val) const; 
    bool operator >= (float val) const; 
    bool operator >= (const FixedNum &val) const; 
    bool operator <= (float val) const; 
    bool operator <= (const FixedNum &val) const; 
    
    const FixedNum & operator += (float val); 
    const FixedNum & operator += (const FixedNum &val); 
    const FixedNum & operator -= (float val); 
    const FixedNum & operator -= (const FixedNum &val); 
    const FixedNum & operator *= (float val); 
    const FixedNum & operator *= (const FixedNum &val); 
    
    FixedNum operator - () const; 
    FixedNum operator + (float val) const; 
    FixedNum operator + (const FixedNum & val) const; 
    FixedNum operator - (float val) const; 
    FixedNum operator - (const FixedNum & val) const; 
    FixedNum operator * (float val) const; 
    FixedNum operator * (const FixedNum & val) const; 
    
    FixedNum contract() const; 
    
    operator float() const; 
    
}; 


    
}


#endif
