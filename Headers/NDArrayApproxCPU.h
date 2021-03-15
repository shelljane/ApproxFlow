#ifndef __APPROXFLOW_NDARRAYAPPROXCPU__
#define __APPROXFLOW_NDARRAYAPPROXCPU__

#include "Common.h"

#include "Range.h"

#include "NDArrayFloatCPU.h"

namespace ApproxFlow
{
// TODO Padding is WRONG for stride > 1
template<typename Scalar>
class NDArrayApproxCPU
{
public:
    
    typedef int AccumType; 
    typedef unsigned AccumTypeUnsigned; 
    typedef unsigned short AddType; 
//     typedef int AddType; 
    typedef float FloatType; 
    typedef NDArrayFloatCPU<float> NDArrayFloat; 
    
    static void _loadLUT(const std::string & filename); 
    static void _loadLUTs(const std::vector<std::string> & filenames); 
    static void _switchLUT(size_t index) {_LUT = _LUTs[index]; } 
    
private: 
    
    FloatType _S; 
    AddType _Z; 
    
    size_t _size; 
    
    std::vector<size_t> _shape; 
    std::vector<Scalar> _array; 
    
    static std::vector<std::vector<AddType>> _LUT; 
    static std::vector<std::vector<std::vector<AddType>>> _LUTs; 
    
    bool _checkRange(const std::vector<size_t> &index) const;  
    bool _checkRange(const std::vector<Range> &index) const;  
    size_t _getIndex(const std::vector<size_t> &index) const;  
    
    static size_t _mulUp(const std::vector<size_t> &shape);  
    template<typename FixedNum>
    static FixedNum _toFixedNum(float r); 
    template<typename FixedNum, typename RaisedNum>
    static FixedNum _addFixedNum(FixedNum a, FixedNum b);
    template<typename FixedNum, typename RaisedNum>
    static FixedNum _subtractFixedNum(FixedNum a, FixedNum b);
    template<typename FixedNum, typename RaisedNum>
    static FixedNum _mulFixedNum(FixedNum a, FixedNum b);
    template<typename FixedNum, typename FixedNumUnsigned>
    static FixedNum _mulSignedFixedNum(FixedNum a, FixedNumUnsigned b);
    template<typename FixedNum, typename FixedNumUnsigned>
    static FixedNum _mulSignedFixedNumShift(FixedNum signedNum, FixedNumUnsigned unsignedNum, unsigned shift); 
    template<typename FixedNum, typename RaisedNum>
    static FixedNum _truncate(RaisedNum a);
    static AddType _approxMul(Scalar a, Scalar b); 
    
public:
    NDArrayApproxCPU();
    
    NDArrayApproxCPU(FloatType S, AddType Z, const std::vector<size_t> &shape);
    NDArrayApproxCPU(FloatType S, AddType Z, const std::vector<size_t> &shape, const std::vector<Scalar> &arr);
    
    NDArrayApproxCPU(const NDArrayApproxCPU<Scalar> &ndarray); 
    NDArrayApproxCPU(NDArrayApproxCPU<Scalar> &&ndarray); 
    
    const std::vector<size_t> &shape() const {return _shape; }
    size_t size() const {return _size; }
    size_t length() const {return _size; }
    
    const NDArrayApproxCPU &resize(const std::vector<size_t> &shape); 
    NDArrayApproxCPU reshape(const std::vector<size_t> &shape) const; 
    
    Scalar &operator [] (size_t index); 
    const Scalar &operator [] (size_t index) const; 
    Scalar &operator [] (const std::vector<size_t> &index); 
    const Scalar &operator [] (const std::vector<size_t> &index) const; 
    NDArrayApproxCPU get(const std::vector<Range> &index) const; 
    void set(const std::vector<Range> &index, const NDArrayApproxCPU &value); 
    void set(const std::vector<Scalar> &value) {_array = value; }
    
    void print() const; 
    NDArrayFloat eval() const; 
    
    FloatType S() const {return _S; } 
    AddType Z() const {return _Z; }; 
    FloatType getS() const {return _S; } 
    AddType getZ() const {return _Z; }; 
    void setS(FloatType S) {_S = S; } 
    void setZ(AddType Z) {_Z = Z; }; 
    
    const NDArrayApproxCPU &operator = (const NDArrayApproxCPU &ndarray); 
    const NDArrayApproxCPU &operator = (NDArrayApproxCPU &&ndarray); 
    
    static NDArrayApproxCPU add(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2); 
    static NDArrayApproxCPU subtract(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2); 
    static NDArrayApproxCPU multiply(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2); 
    static NDArrayApproxCPU matMul(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2); 
    static NDArrayApproxCPU matMAD(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2, const std::vector<AccumType> &bias); 
    static NDArrayApproxCPU matMADReLU(FloatType SReLU, AddType ZReLU, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2, const std::vector<AccumType> &bias); 
    
    NDArrayApproxCPU T() const; 
    NDArrayApproxCPU T(const std::vector<size_t> &axis) const; 
    
    NDArrayApproxCPU ReLU() const; 
//     NDArrayApproxCPU mask01() const; 
    
    std::vector<size_t> argmax() const; 
    size_t posmax() const; 
//     AccumType sum() const; 
//     NDArrayApproxCPU sum(size_t axis = 0) const; 
    
    NDArrayApproxCPU maxPool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding = true) const; 
    NDArrayApproxCPU im2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding = true) const; 
//     NDArrayApproxCPU col2im(const std::vector<size_t> &origin, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding = true) const; 
    std::vector<size_t> sizeIm2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const; 
    std::vector<size_t> sizeIm2Col_ImOnly(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const; 
    std::vector<size_t> sizePool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const; 
    
    // Useless Below
    
    NDArrayApproxCPU addChWise(FloatType S, AddType Z, const NDArrayApproxCPU &ndarray) const; 
//     NDArrayApproxCPU addChWiseRev(const NDArrayApproxCPU &ndarray) const; 
    
    const NDArrayApproxCPU &operator += (const Scalar &value); 
    const NDArrayApproxCPU &operator -= (const Scalar &value); 
    const NDArrayApproxCPU &operator &= (const Scalar &value); 
    const NDArrayApproxCPU &operator *= (const Scalar &value); 
    const NDArrayApproxCPU &operator += (const NDArrayApproxCPU &ndarray); 
    const NDArrayApproxCPU &operator -= (const NDArrayApproxCPU &ndarray); 
    const NDArrayApproxCPU &operator &= (const NDArrayApproxCPU &ndarray); 
    
    NDArrayApproxCPU operator + () const; 
    NDArrayApproxCPU operator - () const; 
    NDArrayApproxCPU operator + (const Scalar &value) const; 
    NDArrayApproxCPU operator - (const Scalar &value) const; 
    NDArrayApproxCPU operator & (const Scalar &value) const; 
    NDArrayApproxCPU operator * (const Scalar &value) const; 
    NDArrayApproxCPU operator + (const NDArrayApproxCPU &ndarray) const; 
    NDArrayApproxCPU operator - (const NDArrayApproxCPU &ndarray) const; 
    NDArrayApproxCPU operator & (const NDArrayApproxCPU &ndarray) const; 
}; 

template<typename Scalar> 
std::vector<std::vector<typename NDArrayApproxCPU<Scalar>::AddType>> NDArrayApproxCPU<Scalar>::_LUT; 

template<typename Scalar> 
std::vector<std::vector<std::vector<typename NDArrayApproxCPU<Scalar>::AddType>>> NDArrayApproxCPU<Scalar>::_LUTs; 

template<typename Scalar>
bool NDArrayApproxCPU<Scalar>::_checkRange(const std::vector<size_t> &index) const
{
    if(index.size() != _shape.size())
    {
        return false; 
    }
    for(size_t idx = 0; idx < index.size(); idx++)
    {
        if(index[idx] >= _shape[idx])
        {
            return false; 
        }
    }
    
    return true; 
}

template<typename Scalar>
bool NDArrayApproxCPU<Scalar>::_checkRange(const std::vector<Range> &index) const
{
    if(index.size() != _shape.size())
    {
        std::cerr << "Size not matched. " << std::endl; 
        return false; 
    }
    for(size_t idx = 0; idx < index.size(); idx++)
    {
        if(index[idx].end() > _shape[idx])
        {
            std::cerr << "Size not matched. " << std::endl; 
            return false; 
        }
    }
    
    return true; 
}

template<typename Scalar>
size_t NDArrayApproxCPU<Scalar>::_getIndex(const std::vector<size_t> &index) const
{
    assert(_checkRange(index)); 
    
    size_t result = index[0]; 
    for(size_t idx = 1; idx < _shape.size(); idx++)
    {
        result = result * _shape[idx] + index[idx]; 
    }
    
    return result; 
}


template<typename Scalar>
size_t NDArrayApproxCPU<Scalar>::_mulUp(const std::vector<size_t> &shape)
{
    size_t size = 1; 
    for(size_t idx = 0; idx < shape.size(); idx++)
    {
        size *= shape[idx]; 
    }
    
    return size; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar>::NDArrayApproxCPU(): _S(), _Z(), _size(0), _shape(), _array() {}

template<typename Scalar>
NDArrayApproxCPU<Scalar>::NDArrayApproxCPU(FloatType S, AddType Z, const std::vector<size_t> &shape): _S(S), _Z(Z), _size(1), _shape(shape), _array() 
{
    for(size_t idx = 0; idx < _shape.size(); idx++)
    {
        _size *= _shape[idx]; 
    }
    _array = std::vector<Scalar>(_size); 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar>::NDArrayApproxCPU(FloatType S, AddType Z, const std::vector<size_t> &shape, const std::vector<Scalar> &arr): _S(S), _Z(Z), _size(1), _shape(shape), _array(arr) 
{
    for(size_t idx = 0; idx < _shape.size(); idx++)
    {
        _size *= _shape[idx]; 
    }
}

template<typename Scalar>
NDArrayApproxCPU<Scalar>::NDArrayApproxCPU(const NDArrayApproxCPU<Scalar> &ndarray): _S(ndarray._S), _Z(ndarray._Z), _size(ndarray._size), _shape(ndarray._shape), _array(ndarray._array) {}

template<typename Scalar>
NDArrayApproxCPU<Scalar>::NDArrayApproxCPU(NDArrayApproxCPU<Scalar> &&ndarray): _S(ndarray._S), _Z(ndarray._Z), _size(ndarray._size), _shape(ndarray._shape), _array(std::move(ndarray._array)) {}


template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::resize(const std::vector<size_t> &shape)
{
    assert(_mulUp(_shape) == _mulUp(shape)); 
    
    _shape = shape; 
    
    return *this; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::reshape(const std::vector<size_t> &shape) const
{
    assert(_mulUp(_shape) == _mulUp(shape)); 
    
    NDArrayApproxCPU<Scalar> result = *this; 
    result.resize(shape);
    
    return result; 
    
}

template<typename Scalar>
Scalar &NDArrayApproxCPU<Scalar>::operator [] (size_t index)
{
    return _array[index]; 
}

template<typename Scalar>
const Scalar &NDArrayApproxCPU<Scalar>::operator [] (size_t index) const
{
    return _array[index]; 
}

template<typename Scalar>
Scalar &NDArrayApproxCPU<Scalar>::operator [] (const std::vector<size_t> &index)
{
    return _array[_getIndex(index)]; 
}

template<typename Scalar>
const Scalar &NDArrayApproxCPU<Scalar>::operator [] (const std::vector<size_t> &index) const
{
    return _array[_getIndex(index)]; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::get(const std::vector<Range> &index) const
{
    assert(_checkRange(index)); 
    
    size_t size = 1; 
    std::vector<size_t> shape(index.size()); 
    for(size_t idx = 0; idx < index.size(); idx++)
    {
        size *= index[idx].size(); 
        shape[idx] = index[idx].size(); 
    }
    
    std::vector<size_t> current(index.size()); 
    for(size_t idx = 0; idx < index.size(); idx++)
    {
        current[idx] = index[idx].begin(); 
    }
    NDArrayApproxCPU<Scalar> result(_S, _Z, shape); 
    for(size_t idx = 0; idx < size; idx++)
    {
        result[idx] = _array[_getIndex(current)]; 
        // Increase the current index
        current[current.size() - 1] += 1; 
        for(size_t jdx = 0; jdx < index.size(); jdx++)
        {
            if(current[index.size() - 1 - jdx] >= index[index.size() - 1 - jdx].end() && index.size() - 1 - jdx > 0)
            {
                current[index.size() - 1 - jdx] = index[index.size() - 1 - jdx].begin();
                current[index.size() - 2 - jdx] += 1; 
            }
        }
    }
    
    return result; 
}

template<typename Scalar>
void NDArrayApproxCPU<Scalar>::set(const std::vector<Range> &index, const NDArrayApproxCPU<Scalar> &value)
{
    assert(_checkRange(index)); 
    
    size_t size = 1; 
    std::vector<size_t> shape(index.size()); 
    for(size_t idx = 0; idx < index.size(); idx++)
    {
        size *= index[idx].size(); 
        shape[idx] = index[idx].size(); 
    }
    
    std::vector<size_t> current(index.size()); 
    for(size_t idx = 0; idx < index.size(); idx++)
    {
        current[idx] = index[idx].begin(); 
    }
    for(size_t idx = 0; idx < size; idx++)
    {
        _array[_getIndex(current)] = value[idx]; 
        // Increase the current index
        current[current.size() - 1] += 1; 
        for(size_t jdx = 0; jdx < index.size(); jdx++)
        {
            if(current[index.size() - 1 - jdx] >= index[index.size() - 1 - jdx].end() && index.size() - 1 - jdx > 0)
            {
                current[index.size() - 1 - jdx] = index[index.size() - 1 - jdx].begin();
                current[index.size() - 2 - jdx] += 1; 
            }
        }
    }
}

template<typename Scalar>
void NDArrayApproxCPU<Scalar>::print() const
{
    std::vector<size_t> current(_shape.size(), 0); 
    std::cout << std::endl << "NDArrayApproxCPU: S = " << _S << "; Z = " << _Z << std::endl; 
    for(size_t idx = 0; idx < _size; idx++)
    {
        for(size_t jdx = 0; jdx < _shape.size(); jdx++)
        {
            if(current[_shape.size() - 1 - jdx] == 0)
            {
                std::cout << "["; 
            }
            else
            {
                break; 
            }
        }
        Scalar tmp = _array[_getIndex(current)]; 
        std::cout << std::bitset<(sizeof(Scalar) << 3)>(tmp) << "_(" << static_cast<unsigned>(tmp) << "), "; 
        // Increase the current index
        current[current.size() - 1] += 1; 
        for(size_t jdx = 0; jdx < _shape.size(); jdx++)
        {
            if(current[_shape.size() - 1 - jdx] >= _shape[_shape.size() - 1 - jdx])
            {
                if(_shape.size() - 1 - jdx > 0)
                {
                    current[_shape.size() - 1 - jdx] = 0;
                    current[_shape.size() - 2 - jdx] += 1; 
                    std::cout << "]"; 
                }
                else
                {
                    std::cout << "]" << std::endl; 
                }
            }
            else if(jdx > 0)
            {
                std::cout << std::endl;
                break; 
            }
            else
            {
                break; 
            }
        }
    }
    std::cout << std::endl; 
}

template<typename Scalar>
typename NDArrayApproxCPU<Scalar>::NDArrayFloat NDArrayApproxCPU<Scalar>::eval() const
{
    NDArrayFloat result(_shape); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result[idx] = _S * (static_cast<FloatType>(_array[idx]) - static_cast<FloatType>(_Z)); 
    }
    
    return result; 
}

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator = (const NDArrayApproxCPU<Scalar> &ndarray)
{
    _S = ndarray._S;
    _Z = ndarray._Z; 
    _size = ndarray._size; 
    _shape = ndarray._shape; 
    _array = ndarray._array; 
    
    return *this; 
}

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator = (NDArrayApproxCPU &&ndarray)
{
    _S = ndarray._S;
    _Z = ndarray._Z; 
    _size = ndarray._size; 
    _shape = ndarray._shape; 
    _array = std::move(ndarray._array); 
    
    return *this; 
}

//NOTE: every signed type should be dealed specially
template<typename Scalar>
template<typename FixedNum>
FixedNum NDArrayApproxCPU<Scalar>::_toFixedNum(float r)
{
    assert(r >= 0.0 && r <= 1.0); 
//     if(!(r >= 0.0 && r <= 1.0))
//     {
//         std::cout << "NOT OK: r = " << r << std::endl; 
//     }
//     std::cout << "r: " << r << " ; casted: " << (unsigned)(static_cast<FixedNum>(~0)) << std::endl; 
    
    return round(r * (static_cast<FixedNum>(~0))); 
}


template<typename Scalar>
template<typename FixedNum, typename RaisedNum>
FixedNum NDArrayApproxCPU<Scalar>::_truncate(RaisedNum a)
{
    FixedNum tmp = ((1 << (8 * sizeof(FixedNum))) - 1); 
    if(a > tmp)
    {
        return tmp; 
    }
    return a; 
}

template<typename Scalar>
void NDArrayApproxCPU<Scalar>::_loadLUT(const std::string & filename)
{
    _LUT.clear(); 
    size_t size = (1 << (8 *sizeof(Scalar))); 
    std::ifstream fin(filename); 
    if(!fin)
    {
        std::cerr << "ERROR: failed to open the file: " << filename << std::endl; 
        exit(1); 
    }
    std::cout << "Loading the Look Up Table. " << std::endl; 
    std::cout << "\t" << std::endl;
    for(size_t idx = 0; idx < size; idx++)
    {
        std::cout << idx << "\t"; 
    }
    std::cout << std::endl; 
    for(size_t idx = 0; idx < size; idx++)
    {
        _LUT.push_back(std::vector<AddType>()); 
        for(size_t jdx= 0; jdx < size; jdx++)
        {
            long tmp; 
            fin >> tmp; 
            _LUT[idx].push_back(tmp); 
            std::cout << tmp << "\t"; 
        }
        std::cout << std::endl; 
    }
}

template<typename Scalar>
void NDArrayApproxCPU<Scalar>::_loadLUTs(const std::vector<std::string> & filenames)
{
    std::vector<std::vector<AddType>> backup = _LUT; 
    _LUTs.clear(); 
    for(size_t idx = 0; idx < filenames.size(); idx++)
    {
        _loadLUT(filenames[idx]); 
        _LUTs.push_back(_LUT); 
    }
    _LUT = backup; 
}

template<typename Scalar>
typename NDArrayApproxCPU<Scalar>::AddType NDArrayApproxCPU<Scalar>::_approxMul(Scalar a, Scalar b)
{
    assert(_LUT.size() > 0);
    return _LUT[a][b]; 
}

template<typename Scalar>
template<typename FixedNum, typename RaisedNum>
FixedNum NDArrayApproxCPU<Scalar>::_addFixedNum(FixedNum a, FixedNum b)
{
    RaisedNum raised = static_cast<RaisedNum>(a) + static_cast<RaisedNum>(b); 
    FixedNum fixed = ((1 << (8 * sizeof(FixedNum))) - 1); 
    if(raised <= fixed)
    {
        fixed = raised; 
    }
    return fixed; 
}

template<typename Scalar>
template<typename FixedNum, typename RaisedNum>
FixedNum NDArrayApproxCPU<Scalar>::_subtractFixedNum(FixedNum a, FixedNum b)
{
    RaisedNum raised = static_cast<RaisedNum>(a) - static_cast<RaisedNum>(b); 
    
    if(b > a)
    {
        raised = 0; 
    }
    
    return raised; 
}

template<typename Scalar>
template<typename FixedNum, typename RaisedNum>
FixedNum NDArrayApproxCPU<Scalar>::_mulFixedNum(FixedNum a, FixedNum b)
{
    RaisedNum raised; 
    raised = static_cast<RaisedNum>(a) * static_cast<RaisedNum>(b); 
    FixedNum fixed = (raised >> (8 * sizeof(FixedNum))); 
    if((raised & (1 << ((8 * sizeof(FixedNum)) - 1))) > 0)
    {
        fixed += 1; 
    }
    return fixed; 
}

template<typename Scalar>
template<typename FixedNum, typename FixedNumUnsigned>
FixedNum NDArrayApproxCPU<Scalar>::_mulSignedFixedNum(FixedNum signedNum, FixedNumUnsigned unsignedNum)
{
    long tmp; 
    unsigned long tmp2; 
    if(signedNum < 0)
    {
        tmp = -signedNum; 
        tmp *= unsignedNum; 
        if((tmp & (1 << ((8 * sizeof(FixedNum)) - 1))) > 0)
        {
            tmp >>= (8 * sizeof(FixedNum)); 
            tmp += 1; 
        }
        else
        {
            tmp >>= (8 * sizeof(FixedNum)); 
        }
        tmp = -tmp; 
    }
    else 
    {
        tmp = signedNum; 
        tmp *= unsignedNum; 
//         std::cout << "tmp product: " << tmp << std::endl; 
        if((tmp & (1 << ((8 * sizeof(FixedNum)) - 1))) > 0)
        {
            tmp >>= (8 * sizeof(FixedNum)); 
            tmp += 1; 
        }
        else
        {
            tmp >>= (8 * sizeof(FixedNum)); 
        }
//         std::cout << "tmp product: " << tmp << std::endl; 
    }
    
    return tmp; 
}

//TODO: wrong
template<typename Scalar>
template<typename FixedNum, typename FixedNumUnsigned>
FixedNum NDArrayApproxCPU<Scalar>::_mulSignedFixedNumShift(FixedNum signedNum, FixedNumUnsigned unsignedNum, unsigned shift)
{
    long tmp; 
    unsigned long tmp2; 
    if(signedNum < 0)
    {
        tmp = -signedNum; 
        tmp2 = tmp; 
        tmp2 *= static_cast<unsigned long>(unsignedNum); 
        if((tmp2 & (1 << ((8 * sizeof(FixedNum)) - 1))) > 0)
        {
            tmp2 >>= (8 * sizeof(FixedNum)); 
            tmp2 += 1; 
        }
        else
        {
            tmp2 >>= (8 * sizeof(FixedNum)); 
        }
        tmp = (tmp2 >> shift);
        tmp = -tmp; 
    }
    else // greater than 0, TODO: WRONG, overflow!!!
    {
        tmp2 = signedNum; 
        tmp2 *= static_cast<unsigned long>(unsignedNum); 
//         std::cout << "tmp product: " << tmp << std::endl; 
        if((tmp2 & (1 << ((8 * sizeof(FixedNum)) - 1))) > 0)
        {
            tmp2 >>= (8 * sizeof(FixedNum)); 
            tmp2 += 1; 
        }
        else
        {
            tmp2 >>= (8 * sizeof(FixedNum)); 
        }
//         std::cout << "tmp product: " << tmp << std::endl; 
        tmp = (tmp2 >> shift); 
    }
    
    return tmp; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::add(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2)
{
    assert(arr1._size == arr2._size); 
    assert(S >= arr1._S && S >= arr2._S); 
    
    NDArrayApproxCPU<Scalar> result(S, Z, arr1._shape); 
    
    Scalar scale_1_3_a = _toFixedNum<Scalar>(arr1._S / S); 
    Scalar scale_2_3_a = _toFixedNum<Scalar>(arr2._S / S); 
    AddType scale_1_3_b = _toFixedNum<AddType>(arr1._S / S); 
    AddType scale_2_3_b = _toFixedNum<AddType>(arr2._S / S); 
//     std::cout << "Scales: " << static_cast<unsigned>(scale_1_3_a) << ", " << static_cast<unsigned>(scale_2_3_a) << ", " << scale_1_3_b << ", " << scale_2_3_b << std::endl; 
    
    //TODO: 解决所有定点数乘法问题，应该是把位数翻倍后，乘在一起，然后砍掉后面一半
    for(size_t idx = 0; idx < arr1._size; idx++)
    {   
        AddType tmp = Z; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp += _mulFixedNum<Scalar, AddType>(scale_1_3_a, arr1._array[idx]); 
//         std::cout << "result: " << (unsigned)_mulFixedNum<Scalar, AddType>(scale_1_3_a, arr1._array[idx]) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp += _mulFixedNum<Scalar, AddType>(scale_2_3_a, arr2._array[idx]); 
//         std::cout << "result: " << (unsigned)_mulFixedNum<Scalar, AddType>(scale_2_3_a, arr2._array[idx]) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp = _subtractFixedNum<AddType, AddType>(tmp, _mulFixedNum<AddType, AccumType>(scale_1_3_b, arr1._Z)); 
//         std::cout << "result: " << _mulFixedNum<AddType, AccumType>(scale_1_3_b, arr1._Z) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp = _subtractFixedNum<AddType, AddType>(tmp, _mulFixedNum<AddType, AccumType>(scale_2_3_b, arr2._Z)); 
//         std::cout << "result: " << _mulFixedNum<AddType, AccumType>(scale_2_3_b, arr2._Z) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        result._array[idx] = _truncate<Scalar, AddType>(tmp); 
        
    }
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::subtract(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2)
{
    assert(arr1._size == arr2._size); 
    assert(S >= arr1._S && S >= arr2._S); 
    
    NDArrayApproxCPU<Scalar> result(S, Z, arr1._shape); 
    
    Scalar scale_1_3_a = _toFixedNum<Scalar>(arr1._S / S); 
    Scalar scale_2_3_a = _toFixedNum<Scalar>(arr2._S / S); 
    AddType scale_1_3_b = _toFixedNum<AddType>(arr1._S / S); 
    AddType scale_2_3_b = _toFixedNum<AddType>(arr2._S / S); 
    
    for(size_t idx = 0; idx < arr1._size; idx++)
    {
//         std::cout << "Scales: " << static_cast<unsigned>(scale_1_3_a) << ", " << static_cast<unsigned>(scale_2_3_a) << ", " << scale_1_3_b << ", " << scale_2_3_b << std::endl; 
        AddType tmp = Z; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp += _mulFixedNum<Scalar, AddType>(scale_1_3_a, arr1._array[idx]); 
//         std::cout << "result: " << (unsigned)_mulFixedNum<Scalar, AddType>(scale_1_3_a, arr1._array[idx]) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp += _mulFixedNum<AddType, AccumType>(scale_2_3_b, arr2._Z); 
//         std::cout << "result: " << (unsigned)_mulFixedNum<AddType, AccumType>(scale_2_3_b, arr2._Z) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp = _subtractFixedNum<AddType, AddType>(tmp, _mulFixedNum<Scalar, AddType>(scale_2_3_a, arr2._array[idx])); 
//         std::cout << "arr2: " << (unsigned)arr2._array[idx] << std::endl; 
//         std::cout << "result: " << (unsigned)_mulFixedNum<Scalar, AddType>(scale_2_3_a, arr2._array[idx]) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        tmp = _subtractFixedNum<AddType, AddType>(tmp, _mulFixedNum<AddType, AccumType>(scale_1_3_b, arr1._Z)); 
//         std::cout << "result: " << (unsigned)_mulFixedNum<AddType, AccumType>(scale_1_3_b, arr1._Z) << std::endl; 
//         std::cout << "tmp: " << tmp << std::endl; 
        result._array[idx] = _truncate<Scalar, AddType>(tmp);
    }
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::multiply(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2)
{
    assert(arr1._size == arr2._size); 
    assert(S >= arr1._S && S >= arr2._S); 
    
    NDArrayApproxCPU<Scalar> result(S, Z, arr1._shape); 
    
    AccumTypeUnsigned scale = _toFixedNum<AccumTypeUnsigned>(arr1._S * arr2._S / S); 
//     std::cout << "scale: " << scale << std::endl; 
    for(size_t idx = 0; idx < arr1._size; idx++)
    {
        AccumType tmp1 = static_cast<AccumType>(arr1._array[idx]) * static_cast<AccumType>(arr2._array[idx]); 
//         std::cout << "tmp1: " << tmp1 << std::endl; 
        tmp1 += static_cast<AccumType>(arr1._Z) * static_cast<AccumType>(arr2._Z); 
//         std::cout << "tmp1: " << tmp1 << std::endl; 
        tmp1 -= static_cast<AccumType>(arr1._array[idx]) * static_cast<AccumType>(arr2._Z); 
//         std::cout << "tmp1: " << tmp1 << std::endl; 
        tmp1 -= static_cast<AccumType>(arr2._array[idx]) * static_cast<AccumType>(arr1._Z); 
//         std::cout << "tmp1: " << tmp1 << std::endl; 
        AccumType tmp2 = Z; 
//         std::cout << "tmp2: " << tmp2 << std::endl; 
        tmp2 += _mulSignedFixedNum<AccumType, AccumTypeUnsigned>(tmp1, scale); 
//         std::cout << "tmp2: " << tmp2 << std::endl; 
        result._array[idx] = tmp2; 
    }
    
    return result; 
}

//TODO: Redesign the matMul
template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::matMul(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2)
{
    assert(arr1._shape.size() == 2 && arr2._shape.size() == 2);
    assert(arr1._shape[1] == arr2._shape[0]);
    
    size_t tmpIndex = 0, tmp1 = 0, tmp2 = 0; 
    AccumType tmpSum1, tmpSum2, tmpSum3, tmpSum, tmpAccum; 
    float scale_f = arr1._S * arr2._S / S; 
    unsigned shift = 0; 
    while(scale_f < 0.5)
    {
        scale_f *= 2; 
        shift++; 
    }
    if(scale_f >= 1.0 - __APPROXFLOW_EPSILON__)
    {
        scale_f = 1.0; 
    }
    AccumTypeUnsigned scale =  _toFixedNum<AccumTypeUnsigned>(scale_f); 
    
//     std::cout << "shift: " << shift << std::endl; 
//     std::cout << "scale: " << scale << std::endl; 
//     std::cout << "scale_f: " << scale_f << std::endl; 
    
    NDArrayApproxCPU<Scalar> tmp = arr2.T(); 
    NDArrayApproxCPU<Scalar> result(S, Z, {arr1._shape[0], tmp._shape[0]}); 
    
    for(size_t idx = 0; idx < arr1._shape[0]; idx++)
    {
        for(size_t jdx = 0; jdx < tmp._shape[0]; jdx++)
        {
            tmp1 = idx * arr1._shape[1]; 
            tmp2 = jdx * tmp._shape[1]; 
            tmpAccum = Z; 
            tmpSum = static_cast<AccumType>(arr1._shape[1]) * static_cast<AccumType>(arr1._Z) * static_cast<AccumType>(tmp._Z); 
//             std::cout << "tmpSum: " << tmpSum << std::endl; 
            tmpSum1 = tmpSum2 = tmpSum3 = 0; 
            for(size_t kdx = 0; kdx < tmp._shape[1]; kdx++, tmp1++, tmp2++)
            {
                tmpSum1 += static_cast<AccumType>(arr1._array[tmp1]); 
                tmpSum2 += static_cast<AccumType>(tmp._array[tmp2]); 
                tmpSum3 += _approxMul(arr1._array[tmp1], tmp._array[tmp2]); 
//                 tmpSum3 += static_cast<AccumType>(arr1._array[tmp1]) * static_cast<AccumType>(tmp._array[tmp2]); 
            }
//             std::cout << "tmpSum1: " << tmpSum1 << std::endl; 
//             std::cout << "tmpSum2: " << tmpSum2 << std::endl; 
//             std::cout << "tmpSum3: " << tmpSum3 << std::endl; 
            tmpSum = tmpSum - arr1._Z * tmpSum2 - tmp._Z * tmpSum1 + tmpSum3; 
//             std::cout << "tmpSum: " << tmpSum << std::endl; 
//             std::cout << "tmpSumX: " << _mulSignedFixedNumShift<AccumType, AccumTypeUnsigned>(tmpSum, scale, shift) << std::endl; 
            tmpAccum += _mulSignedFixedNumShift<AccumType, AccumTypeUnsigned>(tmpSum, scale, shift); 
            if(tmpAccum < 0)
            {
                tmpAccum = 0; 
            }
            else if(tmpAccum > (1 << (8 * sizeof(Scalar))) - 1)
            {
                tmpAccum = (1 << (8 * sizeof(Scalar))) - 1; 
            }
//             std::cout << "tmpAccum: " << tmpAccum << std::endl; 
//             std::cout << std::endl; 
            
            result._array[tmpIndex++] = tmpAccum; 
        }
    }
    
    return result; 
}

//ATTENTION: Make sure that the S and Z of the bias are S1*S2 and 0, respectively
template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::matMAD(FloatType S, AddType Z, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2, const std::vector<AccumType> &bias)
{
    assert(arr1._shape.size() == 2 && arr2._shape.size() == 2);
    assert(arr1._shape[1] == arr2._shape[0]);
    assert(bias.size() == arr2._shape[1]); 
    
    size_t tmpIndex = 0, tmp1 = 0, tmp2 = 0; 
    AccumType tmpSum1, tmpSum2, tmpSum3, tmpSum, tmpAccum; 
    float scale_f = arr1._S * arr2._S / S; 
    unsigned shift = 0; 
    while(scale_f < 0.5)
    {
        scale_f *= 2; 
        shift++; 
    }
    if(scale_f >= 1.0 - __APPROXFLOW_EPSILON__)
    {
        scale_f = 1.0; 
    }
    AccumTypeUnsigned scale =  _toFixedNum<AccumTypeUnsigned>(scale_f); 
    
//     std::cout << "shift: " << shift << std::endl; 
//     std::cout << "scale: " << scale << std::endl; 
//     std::cout << "scale_f: " << scale_f << std::endl; 
    
    NDArrayApproxCPU<Scalar> tmp = arr2.T(); 
    NDArrayApproxCPU<Scalar> result(S, Z, {arr1._shape[0], tmp._shape[0]}); 
    
    for(size_t idx = 0; idx < arr1._shape[0]; idx++)
    {
        for(size_t jdx = 0; jdx < tmp._shape[0]; jdx++)
        {
            tmp1 = idx * arr1._shape[1]; 
            tmp2 = jdx * tmp._shape[1]; 
            tmpAccum = Z; 
            tmpSum = static_cast<AccumType>(arr1._shape[1]) * static_cast<AccumType>(arr1._Z) * static_cast<AccumType>(tmp._Z); 
//             std::cout << "tmpSum: " << tmpSum << std::endl; 
            tmpSum1 = tmpSum2 = tmpSum3 = 0; 
            for(size_t kdx = 0; kdx < tmp._shape[1]; kdx++, tmp1++, tmp2++)
            {
                tmpSum1 += static_cast<AccumType>(arr1._array[tmp1]); 
                tmpSum2 += static_cast<AccumType>(tmp._array[tmp2]); 
                tmpSum3 += _approxMul(arr1._array[tmp1], tmp._array[tmp2]); 
//                 tmpSum3 += static_cast<AccumType>(arr1._array[tmp1]) * static_cast<AccumType>(tmp._array[tmp2]); 
            }
//             std::cout << "tmpSum1: " << tmpSum1 << std::endl; 
//             std::cout << "tmpSum2: " << tmpSum2 << std::endl; 
//             std::cout << "tmpSum3: " << tmpSum3 << std::endl; 
            tmpSum = tmpSum - arr1._Z * tmpSum2 - tmp._Z * tmpSum1 + tmpSum3 + bias[jdx]; 
//             std::cout << "tmpSum: " << tmpSum << std::endl; 
//             std::cout << "tmpSumX: " << _mulSignedFixedNumShift<AccumType, AccumTypeUnsigned>(tmpSum, scale, shift) << std::endl; 
            tmpAccum += _mulSignedFixedNumShift<AccumType, AccumTypeUnsigned>(tmpSum, scale, shift); 
            if(tmpAccum < 0)
            {
                tmpAccum = 0; 
            }
            else if(tmpAccum > (1 << (8 * sizeof(Scalar))) - 1)
            {
                tmpAccum = (1 << (8 * sizeof(Scalar))) - 1; 
            }
//             std::cout << "tmpAccum: " << tmpAccum << std::endl; 
//             std::cout << std::endl; 
            
            result._array[tmpIndex++] = tmpAccum; 
        }
    }
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::matMADReLU(FloatType SReLU, AddType ZReLU, const NDArrayApproxCPU &arr1, const NDArrayApproxCPU &arr2, const std::vector<AccumType> &bias)
{
    assert(arr1._shape.size() == 2 && arr2._shape.size() == 2);
    if(!(arr1._shape[1] == arr2._shape[0]))
    {
        ApproxFlow::print(arr1._shape); 
        ApproxFlow::print(arr2._shape); 
    }
    assert(arr1._shape[1] == arr2._shape[0]);
    assert(bias.size() == arr2._shape[1]); 
    
    size_t tmpIndex = 0, tmp1 = 0, tmp2 = 0; 
    AccumType tmpSum1, tmpSum2, tmpSum3, tmpSum, tmpAccum; 
    float scale_f = arr1._S * arr2._S / SReLU; 
    unsigned shift = 0; 
    while(scale_f < 0.5)
    {
        scale_f *= 2; 
        shift++; 
    }
    if(scale_f >= 1.0 - __APPROXFLOW_EPSILON__)
    {
        scale_f = 1.0; 
    }
    AccumTypeUnsigned scale =  _toFixedNum<AccumTypeUnsigned>(scale_f); 
    
//     std::cout << "shift: " << shift << std::endl; 
//     std::cout << "scale: " << scale << std::endl; 
//     std::cout << "scale_f: " << scale_f << std::endl; 
    
    NDArrayApproxCPU<Scalar> tmp = arr2.T(); 
    NDArrayApproxCPU<Scalar> result(SReLU, ZReLU, {arr1._shape[0], tmp._shape[0]}); 
    
    for(size_t idx = 0; idx < arr1._shape[0]; idx++)
    {
        for(size_t jdx = 0; jdx < tmp._shape[0]; jdx++)
        {
            tmp1 = idx * arr1._shape[1]; 
            tmp2 = jdx * tmp._shape[1]; 
//             tmpAccum = Z; 
            tmpSum = static_cast<AccumType>(arr1._shape[1]) * static_cast<AccumType>(arr1._Z) * static_cast<AccumType>(tmp._Z); 
//             std::cout << "tmpSum: " << tmpSum << std::endl; 
            tmpSum1 = tmpSum2 = tmpSum3 = 0; 
            for(size_t kdx = 0; kdx < tmp._shape[1]; kdx++, tmp1++, tmp2++)
            {
                tmpSum1 += static_cast<AccumType>(arr1._array[tmp1]); 
                tmpSum2 += static_cast<AccumType>(tmp._array[tmp2]); 
                tmpSum3 += _approxMul(arr1._array[tmp1], tmp._array[tmp2]); 
//                 tmpSum3 += static_cast<AccumType>(arr1._array[tmp1]) * static_cast<AccumType>(tmp._array[tmp2]); 
            }
//             std::cout << "tmpSum1: " << tmpSum1 << std::endl; 
//             std::cout << "tmpSum2: " << tmpSum2 << std::endl; 
//             std::cout << "tmpSum3: " << tmpSum3 << std::endl; 
            tmpSum = tmpSum - arr1._Z * tmpSum2 - tmp._Z * tmpSum1 + tmpSum3 + bias[jdx]; 
//             std::cout << "tmpSum: " << tmpSum << std::endl; 
//             std::cout << "tmpSumX: " << _mulSignedFixedNumShift<AccumType, AccumTypeUnsigned>(tmpSum, scale, shift) << std::endl; 
            tmpAccum = _mulSignedFixedNumShift<AccumType, AccumTypeUnsigned>(tmpSum, scale, shift); 
            if(tmpAccum < 0)
            {
                tmpAccum = 0; 
            }
            else if(tmpAccum > (1 << (8 * sizeof(Scalar))) - 1)
            {
                tmpAccum = (1 << (8 * sizeof(Scalar))) - 1; 
            }
//             std::cout << "tmpAccum: " << tmpAccum << std::endl; 
//             std::cout << std::endl; 
            
            result._array[tmpIndex++] = tmpAccum; 
        }
    }
    
    return result; 
}


template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::T() const
{
    assert(_shape.size() == 2); 
    
    constexpr size_t SizeBlock = __APPROXFLOW_BLOCK_SIZE_TRANSPOSE__; 
    const size_t nBlocksX = _shape[0] / SizeBlock, nBlocksY = _shape[1] / SizeBlock; 
    size_t index1, index2, indexpre1, indexpre2; 
    
    NDArrayApproxCPU<Scalar> result(_S, _Z, {_shape[1], _shape[0]}); 
    
    for(size_t idx = 0; idx < nBlocksX; idx++)
    {
        for(size_t jdx = 0; jdx < nBlocksY; jdx++)
        {
            indexpre1 = idx*SizeBlock * _shape[1] + jdx*SizeBlock;
            indexpre2 = jdx*SizeBlock * _shape[0] + idx*SizeBlock;
            for(size_t kdx = 0; kdx < SizeBlock; kdx++)
            {
                index1 = indexpre1 + kdx * _shape[1]; 
                index2 = indexpre2 + kdx; 
                for(size_t ldx = 0; ldx < SizeBlock; ldx++)
                {
                    result._array[index2] = _array[(index1++)]; 
                    index2 += _shape[0]; 
//                     result._array[index2 + kdx + ldx * _shape[0]] = _array[index1 + ldx + kdx * _shape[1]]; 
                }
            }
        }
        for(size_t jdx = nBlocksY * SizeBlock; jdx < _shape[1]; jdx++)
        {
            index1 = idx*SizeBlock * _shape[1] + jdx; 
            index2 = jdx * _shape[0] + idx*SizeBlock; 
            for(size_t kdx = 0; kdx < SizeBlock; kdx++)
            {
                result._array[index2++] = _array[index1]; 
                index1 += _shape[1]; 
            }
        }
    }
    for(size_t idx = nBlocksX * SizeBlock; idx < _shape[0]; idx++)
    {
        for(size_t jdx = 0; jdx < nBlocksY; jdx++)
        {
            index1 = idx * _shape[1] + jdx*SizeBlock; 
            index2 = jdx*SizeBlock * _shape[0] + idx; 
            for(size_t ldx = 0; ldx < SizeBlock; ldx++)
            {
                result._array[index2] = _array[index1++];
                index2 += _shape[0]; 
            }
        }
        for(size_t jdx = nBlocksY * SizeBlock; jdx < _shape[1]; jdx++)
        {
            index1 = idx * _shape[1] + jdx; 
            index2 = jdx * _shape[0] + idx; 
            result._array[index2] = _array[index1]; 
        }
    }
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::T(const std::vector<size_t> &axis) const
{
    assert(axis.size() == _shape.size()); 
    
    std::vector<size_t> shape(_shape.size()); 
    for(size_t idx = 0; idx < _shape.size(); idx++)
    {
        shape[idx] = _shape[axis[idx]]; 
    }
    
    NDArrayApproxCPU<Scalar> result(_S, _Z, shape); 
    
    std::vector<size_t> current(shape.size(), 0); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result._array[idx] = _array[_getIndex(current)]; 
        // Increase the current index
        current[axis[current.size() - 1]] += 1; 
        for(size_t jdx = 0; jdx < shape.size(); jdx++)
        {
            if(current[axis[current.size() - 1 - jdx]] >= shape[current.size() - 1 - jdx] && current.size() - 1 - jdx > 0)
            {
                current[axis[current.size() - 1 - jdx]] = 0;
                current[axis[current.size() - 2 - jdx]] += 1; 
            }
        }
    }
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::ReLU() const
{
    NDArrayApproxCPU<Scalar> result(_S, _Z, _shape); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result._array[idx] = static_cast<AddType>(result._array[idx]) > _Z ? result._array[idx] : _Z; 
    }
    
    return result; 
}

template<typename Scalar>
std::vector<size_t> NDArrayApproxCPU<Scalar>::argmax() const
{
    std::vector<size_t> result; 
    
    Scalar tmpMax = _array[0]; 
    std::vector<size_t> current(_shape.size(), 0); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        if(tmpMax < _array[idx])
        {
            result = current; 
            tmpMax = _array[idx]; 
        }
        // Increase the current index
        current[current.size() - 1] += 1; 
        for(size_t jdx = 0; jdx < _shape.size(); jdx++)
        {
            if(current[current.size() - 1 - jdx] >= _shape[current.size() - 1 - jdx] && current.size() - 1 - jdx > 0)
            {
                current[current.size() - 1 - jdx] = 0;
                current[current.size() - 2 - jdx] += 1; 
            }
        }
    }
    
    return result; 
}

template<typename Scalar>
size_t NDArrayApproxCPU<Scalar>::posmax() const
{
    size_t tmpPos = 0; 
    Scalar tmpMax = _array[0]; 
    for(size_t idx = 0; idx < _size; idx++)
    {
        if(_array[idx] > tmpMax)
        {
            tmpMax = _array[idx]; 
            tmpPos = idx; 
        }
    }
    
    return tmpPos; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::addChWise(FloatType S, AddType Z, const NDArrayApproxCPU &ndarray) const
{
    assert(ndarray._shape[0] == _shape[2]); 
    assert(ndarray._shape.size() == 1); 
    assert(_shape.size() == 3); 
    
    NDArrayFloatCPU<Scalar> result = *this; 
    
    size_t index = 0; 
    for(size_t idx = 0; idx < _size; idx++)
    {
        Scalar scale_1_3_a = _toFixedNum<Scalar>(_S / S); 
        Scalar scale_2_3_a = _toFixedNum<Scalar>(ndarray._S / S); 
        AddType scale_1_3_b = _toFixedNum<AddType>(_S / S); 
        AddType scale_2_3_b = _toFixedNum<AddType>(ndarray._S / S); 
        AddType tmp = Z; 
        tmp += _mulFixedNum<AddType, AccumType>(scale_1_3_a, _array[idx]); 
        tmp += _mulFixedNum<AddType, AccumType>(scale_2_3_a, ndarray._array[index]); 
        tmp = _subtractFixedNum<AddType, AddType>(tmp, _mulFixedNum<AddType, AccumType>(scale_1_3_b, _Z)); 
        tmp = _subtractFixedNum<AddType, AddType>(tmp, _mulFixedNum<AddType, AccumType>(scale_2_3_b, ndarray._Z)); 
        result._array[idx] = tmp; 
        index = (index + 1) % ndarray._shape[0]; 
    }
    
    return result; 
}


template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::maxPool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const 
{
    assert(_shape.size() == 3); 
    assert(size.size() == 2); 
    assert(stride.size() == 2); 
    
    std::vector<size_t> shape(2); 
    if(padding)
    {
        shape[0] = (_shape[0] + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] + stride[1] - 1) / stride[1]; 
    }
    else
    {
        assert((_shape[0] - size[0] + 1) % stride[0] == 0 && (_shape[1] - size[1] + 1) % stride[1] == 0); 
        
        shape[0] = (_shape[0] - size[0] + 1 + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] - size[1] + 1 + stride[0] - 1) / stride[1]; 
    }
    
    NDArrayApproxCPU<Scalar> result(_S, _Z, {shape[0], shape[1], _shape[2]}); 
    
    if(padding)
    {
        size_t tmp0 = shape[0] * stride[0] + (size[0] - stride[0]) - _shape[0]; 
		size_t tmp1 = shape[1] * stride[1] + (size[1] - stride[1]) - _shape[1]; 
        size_t padX = (tmp0 / 2) > 0 ? (tmp0 / 2) : 0;
        size_t padY = (tmp1 / 2) > 0 ? (tmp1 / 2) : 0;
        size_t index1, index2; 
        for(size_t idx = 0; idx < shape[0]; idx++)
        {
            for(size_t jdx = 0; jdx < shape[1]; jdx++)
            {
                std::vector<size_t> maxIndex(_shape[2]); 
                std::vector<Scalar> maxVals(_shape[2], 0); 
                for(size_t kdx = 0; kdx < size[0]; kdx++)
                {
                    for(size_t ldx = 0; ldx < size[1]; ldx++)
                    {
                        for(size_t mdx = 0; mdx < _shape[2]; mdx++)
                        {
                            if(idx*stride[0] + kdx < padX || idx*stride[0] + kdx >= _shape[0] + padX
                               || jdx*stride[1] + ldx < padY || jdx*stride[1] + ldx >= _shape[1] + padY)
                            {
                                ; 
                            }
                            else
                            {
                                index1 = ((idx * stride[0] + kdx - padX) * _shape[1] + jdx * stride[1] + ldx - padY) * _shape[2] + mdx; 
                                if(_array[index1] > maxVals[mdx])
                                {
                                    maxIndex[mdx] = index1; 
                                    maxVals[mdx] = _array[index1]; 
                                }
                            }
                        }
                    }
                }
                for(size_t kdx = 0; kdx < _shape[2]; kdx++)
                {
                    index2 = (idx * shape[1] + jdx) * _shape[2] + kdx; 
                    result[index2] = maxVals[kdx]; 
                }
            }
        } 
    }
    else
    {
        size_t index1, index2; 
        for(size_t idx = 0; idx < shape[0]; idx++)
        {
            for(size_t jdx = 0; jdx < shape[1]; jdx++)
            {
                std::vector<Scalar> maxVals(_shape[2], 0); 
                std::vector<size_t> maxIndex(_shape[2]); 
                for(size_t kdx = 0; kdx < size[0]; kdx++)
                {
                    for(size_t ldx = 0; ldx < size[1]; ldx++)
                    {
                        for(size_t mdx = 0; mdx < _shape[2]; mdx++)
                        {
                            index1 = ((idx * stride[0] + kdx) * _shape[1] + jdx * stride[1] + ldx) * _shape[2] + mdx; 
                            if(_array[index1] > maxVals[mdx])
                            {
                                maxIndex[mdx] = index1; 
                                maxVals[mdx] = _array[index1]; 
                            }
                        }
                    }
                }
                for(size_t kdx = 0; kdx < _shape[2]; kdx++)
                {
                    index2 = (idx * shape[1] + jdx) * _shape[2] + kdx; 
                    result[index2] = maxVals[kdx]; 
                }
            }
        } 
    }
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::im2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
{
    assert(_shape.size() == 3); 
    assert(size.size() == 2); 
    assert(stride.size() == 2); 
    
    std::vector<size_t> shape(2); 
    if(padding)
    {
        shape[0] = (_shape[0] + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] + stride[1] - 1) / stride[1]; 
    }
    else
    {
        assert((_shape[0] - size[0] + 1) % stride[0] == 0 && (_shape[1] - size[1] + 1) % stride[1] == 0); 
        
        shape[0] = (_shape[0] - size[0] + 1 + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] - size[1] + 1 + stride[0] - 1) / stride[1]; 
    }
    
    NDArrayApproxCPU<Scalar> result(_S, _Z, {shape[0] * shape[1], size[0] * size[1] * _shape[2]}); 
    
    if(padding)
    {
        size_t tmp0 = shape[0] * stride[0] + (size[0] - stride[0]) - _shape[0]; 
		size_t tmp1 = shape[1] * stride[1] + (size[1] - stride[1]) - _shape[1]; 
        size_t padX = (tmp0 / 2) > 0 ? (tmp0 / 2) : 0;
        size_t padY = (tmp1 / 2) > 0 ? (tmp1 / 2) : 0;
        size_t index1, index2; 
        for(size_t idx = 0; idx < shape[0]; idx++)
        {
            for(size_t jdx = 0; jdx < shape[1]; jdx++)
            {
                for(size_t kdx = 0; kdx < size[0]; kdx++)
                {
                    for(size_t ldx = 0; ldx < size[1]; ldx++)
                    {
                        for(size_t mdx = 0; mdx < _shape[2]; mdx++)
                        {
                            if(idx*stride[0] + kdx < padX || idx*stride[0] + kdx >= _shape[0] + padX
                               || jdx*stride[1] + ldx < padY || jdx*stride[1] + ldx >= _shape[1] + padY)
                            {
                                index2 = ((idx * shape[1] + jdx) * size[0]*size[1] + (kdx * size[1] + ldx)) * _shape[2] + mdx; 
                                result[index2] = 0.0; 
                            }
                            else
                            {
                                index1 = ((idx * stride[0] + kdx - padX) * _shape[1] + jdx * stride[1] + ldx - padY) * _shape[2] + mdx; 
                                index2 = ((idx * shape[1] + jdx) * size[0]*size[1] + (kdx * size[1] + ldx)) * _shape[2] + mdx; 
                                result[index2] = _array[index1]; 
                            }
                        }
                    }
                }
            }
        } 
    }
    else
    {
        size_t begin1, begin2, index1, index2; 
        for(size_t idx = 0; idx < shape[0]; idx++)
        {
            for(size_t jdx = 0; jdx < shape[1]; jdx++)
            {
                begin1 = (idx * stride[0] * _shape[1] + jdx * stride[1]) * _shape[2]; 
                begin2 = (idx * shape[1] + jdx) * size[0]*size[1] * _shape[2]; 
                for(size_t kdx = 0; kdx < size[0]; kdx++)
                {
                    for(size_t ldx = 0; ldx < size[1]; ldx++)
                    {
                        for(size_t mdx = 0; mdx < _shape[2]; mdx++)
                        {
                            index1 = begin1 + ldx * _shape[2] + mdx; 
                            index2 = begin2 + ldx * _shape[2] + mdx; 
                            result[index2] = _array[index1]; 
                        }
                    }
                    begin1 += _shape[1] * _shape[2]; 
                    begin2 += size[1] * _shape[2]; 
                }
            }
        } 
    }
    
    return result; 
}

template<typename Scalar>
std::vector<size_t> NDArrayApproxCPU<Scalar>::sizeIm2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
{
    assert(_shape.size() == 3); 
    assert(size.size() == 2); 
    assert(stride.size() == 2); 
    
    std::vector<size_t> shape(2); 
    if(padding)
    {
        shape[0] = (_shape[0] + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] + stride[1] - 1) / stride[1]; 
    }
    else
    {
        assert((_shape[0] - size[0] + 1) % stride[0] == 0 && (_shape[1] - size[1] + 1) % stride[1] == 0); 
        
        shape[0] = (_shape[0] - size[0] + 1 + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] - size[1] + 1 + stride[0] - 1) / stride[1]; 
    }
    
    return std::vector<size_t>({shape[0] * shape[1], size[0] * size[1] * _shape[2]}); 
}

template<typename Scalar>
std::vector<size_t> NDArrayApproxCPU<Scalar>::sizeIm2Col_ImOnly(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
{
    assert(_shape.size() == 3); 
    assert(size.size() == 2); 
    assert(stride.size() == 2); 
    
    std::vector<size_t> shape(2); 
    if(padding)
    {
        shape[0] = (_shape[0] + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] + stride[1] - 1) / stride[1]; 
    }
    else
    {
        assert((_shape[0] - size[0] + 1) % stride[0] == 0 && (_shape[1] - size[1] + 1) % stride[1] == 0); 
        
        shape[0] = (_shape[0] - size[0] + 1 + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] - size[1] + 1 + stride[0] - 1) / stride[1]; 
    }
    
    return shape; 
}

template<typename Scalar>
std::vector<size_t> NDArrayApproxCPU<Scalar>::sizePool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
{
    assert(_shape.size() == 3); 
    assert(size.size() == 2); 
    assert(stride.size() == 2); 
    
    std::vector<size_t> shape(3); 
    if(padding)
    {
        shape[0] = (_shape[0] + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] + stride[1] - 1) / stride[1]; 
    }
    else
    {
        assert((_shape[0] - size[0] + 1) % stride[0] == 0 && (_shape[1] - size[1] + 1) % stride[1] == 0); 
        
        shape[0] = (_shape[0] - size[0] + 1 + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] - size[1] + 1 + stride[0] - 1) / stride[1]; 
    }
    shape[2] = _shape[2]; 
    
    return shape; 
}


template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator += (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] += value; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator -= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] -= value; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator &= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        AddType tmp = _array[idx] * value; 
        _array[idx] = (tmp >> (sizeof(Scalar) << 3)); 
    }
    
    return *this; 
}

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator *= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        AddType tmp = _array[idx] * value; 
        _array[idx] = (tmp >> (sizeof(Scalar) << 3)); 
    }
    
    return *this; 
}
//TODO

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator += (const NDArrayApproxCPU &ndarray)
{
    assert(_S == ndarray._S && _Z == ndarray._Z); 
    assert(_size == ndarray._size); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] += ndarray._array[idx];
    }
    
    return *this; 
}

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator -= (const NDArrayApproxCPU &ndarray)
{
    assert(_S == ndarray._S && _Z == ndarray._Z); 
    assert(_size == ndarray._size); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] -= ndarray._array[idx];
    }
    
    return *this; 
}

template<typename Scalar>
const NDArrayApproxCPU<Scalar> &NDArrayApproxCPU<Scalar>::operator &= (const NDArrayApproxCPU &ndarray)
{
    assert(_S == ndarray._S && _Z == ndarray._Z); 
    assert(_size == ndarray._size); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        AddType tmp = static_cast<AddType>(_array[idx]) * static_cast<AddType>(ndarray._array[idx]);
        _array[idx] = (tmp >> (sizeof(Scalar) << 3)); 
    }
    
    return *this; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator + () const {return *this; }

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator - () const 
{
    NDArrayApproxCPU<Scalar> result(_S, ~_Z, _shape); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result._array[idx] = (~_array[idx]); 
    }
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator + (const Scalar &value) const
{
    NDArrayApproxCPU<Scalar> result = *this; 
    result += value; 
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator - (const Scalar &value) const
{
    NDArrayApproxCPU<Scalar> result = *this; 
    result -= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator & (const Scalar &value) const
{
    NDArrayApproxCPU<Scalar> result = *this; 
    result &= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator * (const Scalar &value) const
{
    NDArrayApproxCPU<Scalar> result = *this; 
    result *= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator + (const NDArrayApproxCPU &ndarray) const
{
    NDArrayApproxCPU<Scalar> result = *this; 
    result += ndarray; 
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator - (const NDArrayApproxCPU &ndarray) const
{
    NDArrayApproxCPU<Scalar> result = *this; 
    result -= ndarray; 
    
    return result; 
}

template<typename Scalar>
NDArrayApproxCPU<Scalar> NDArrayApproxCPU<Scalar>::operator & (const NDArrayApproxCPU &ndarray) const
{
    NDArrayApproxCPU<Scalar> result = *this; 
    result &= ndarray; 
    
    return result; 
}

}

#endif
