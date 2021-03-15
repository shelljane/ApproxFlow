#ifndef __APPROXFLOW_NDARRAYFLOATCPU__
#define __APPROXFLOW_NDARRAYFLOATCPU__

#include "Common.h"

#include "Range.h"

namespace ApproxFlow
{

template<typename Scalar>
class NDArrayFloatCPU
{
private: 
    size_t _size; 
    std::vector<size_t> _shape; 
    std::vector<Scalar> _array; 
    
    bool _checkRange(const std::vector<size_t> &index) const;  
    bool _checkRange(const std::vector<Range> &index) const;  
    size_t _getIndex(const std::vector<size_t> &index) const;  
    
    static size_t _mulUp(const std::vector<size_t> &shape);  
    
public:
    NDArrayFloatCPU();
    NDArrayFloatCPU(size_t d1);
    NDArrayFloatCPU(size_t d1, size_t d2);
    NDArrayFloatCPU(size_t d1, size_t d2, size_t d3);
    NDArrayFloatCPU(size_t d1, size_t d2, size_t d3, size_t d4);
    NDArrayFloatCPU(size_t d1, size_t d2, size_t d3, size_t d4, size_t d5);
    
    NDArrayFloatCPU(const std::vector<size_t> &shape);
    NDArrayFloatCPU(const std::vector<size_t> &shape, const std::vector<Scalar> &arr);
    
    NDArrayFloatCPU(const NDArrayFloatCPU<Scalar> &ndarray); 
    NDArrayFloatCPU(NDArrayFloatCPU<Scalar> &&ndarray); 
    
    const NDArrayFloatCPU &init(Scalar val); 
    const NDArrayFloatCPU &init(const std::function<Scalar()> &func); 
    
    const std::vector<size_t> &shape() const; 
    size_t size() const;
    size_t length() const;
    
    const NDArrayFloatCPU &resize(const std::vector<size_t> &shape); 
    NDArrayFloatCPU reshape(const std::vector<size_t> &shape) const; 
    
    Scalar &operator [] (size_t index); 
    const Scalar &operator [] (size_t index) const; 
    Scalar &operator [] (const std::vector<size_t> &index); 
    const Scalar &operator [] (const std::vector<size_t> &index) const; 
    NDArrayFloatCPU get(const std::vector<Range> &index) const; 
    void set(const std::vector<Range> &index, const NDArrayFloatCPU &value); 
    
    void print() const; 
    std::vector<float> values() const; 
    NDArrayFloatCPU<float> eval() const; 
    
    const NDArrayFloatCPU &operator = (const NDArrayFloatCPU &ndarray); 
    const NDArrayFloatCPU &operator = (NDArrayFloatCPU &&ndarray); 
    
    const NDArrayFloatCPU &operator += (const Scalar &value); 
    const NDArrayFloatCPU &operator -= (const Scalar &value); 
    const NDArrayFloatCPU &operator &= (const Scalar &value); 
    const NDArrayFloatCPU &operator |= (const Scalar &value); 
    const NDArrayFloatCPU &operator *= (const Scalar &value); 
    const NDArrayFloatCPU &operator /= (const Scalar &value); 
    const NDArrayFloatCPU &operator += (const NDArrayFloatCPU &ndarray); 
    const NDArrayFloatCPU &operator -= (const NDArrayFloatCPU &ndarray); 
    const NDArrayFloatCPU &operator &= (const NDArrayFloatCPU &ndarray); 
    const NDArrayFloatCPU &operator |= (const NDArrayFloatCPU &ndarray); 
    
    NDArrayFloatCPU operator + () const; 
    NDArrayFloatCPU operator - () const; 
    NDArrayFloatCPU operator + (const Scalar &value) const; 
    NDArrayFloatCPU operator - (const Scalar &value) const; 
    NDArrayFloatCPU operator & (const Scalar &value) const; 
    NDArrayFloatCPU operator | (const Scalar &value) const; 
    NDArrayFloatCPU operator * (const Scalar &value) const; 
    NDArrayFloatCPU operator / (const Scalar &value) const; 
    NDArrayFloatCPU operator + (const NDArrayFloatCPU &ndarray) const; 
    NDArrayFloatCPU operator - (const NDArrayFloatCPU &ndarray) const; 
    NDArrayFloatCPU operator & (const NDArrayFloatCPU &ndarray) const; 
    NDArrayFloatCPU operator | (const NDArrayFloatCPU &ndarray) const; 
    NDArrayFloatCPU operator * (const NDArrayFloatCPU &ndarray) const; 
    
    NDArrayFloatCPU T() const; 
    NDArrayFloatCPU T(const std::vector<size_t> &axis) const; 
    
    NDArrayFloatCPU ReLU() const; 
    NDArrayFloatCPU abs() const; 
    NDArrayFloatCPU mask01() const; 
    
    std::vector<size_t> argmax() const; 
    size_t posmax() const; 
    Scalar sum() const; 
    NDArrayFloatCPU sum(size_t axis = 0) const; 
    
    NDArrayFloatCPU addChWise(const NDArrayFloatCPU &ndarray) const; 
    NDArrayFloatCPU addChWiseRev(const NDArrayFloatCPU &ndarray) const; 
    
    NDArrayFloatCPU maxPool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding = true) const; 
    NDArrayFloatCPU im2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding = true) const; 
    NDArrayFloatCPU col2im(const std::vector<size_t> &origin, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding = true) const; 
    std::vector<size_t> sizeIm2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const; 
    std::vector<size_t> sizeIm2Col_ImOnly(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const; 
    std::vector<size_t> sizePool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const; 
}; 


template<typename Scalar>
bool NDArrayFloatCPU<Scalar>::_checkRange(const std::vector<size_t> &index) const
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
bool NDArrayFloatCPU<Scalar>::_checkRange(const std::vector<Range> &index) const
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
size_t NDArrayFloatCPU<Scalar>::_getIndex(const std::vector<size_t> &index) const
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
size_t NDArrayFloatCPU<Scalar>::_mulUp(const std::vector<size_t> &shape)
{
    size_t size = 1; 
    for(size_t idx = 0; idx < shape.size(); idx++)
    {
        size *= shape[idx]; 
    }
    
    return size; 
}


template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(): _size(0), _shape(), _array() {}

template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(size_t d1): _size(d1), _shape({d1}), _array(d1) {}

template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(size_t d1, size_t d2): _size(d1 * d2), _shape({d1, d2}), _array(d1 * d2) {}

template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(size_t d1, size_t d2, size_t d3): _size(d1 * d2 * d3), _shape({d1, d2, d3}), _array(d1 * d2 * d3) {}

template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(size_t d1, size_t d2, size_t d3, size_t d4): _size(d1 * d2 * d3 * d4), _shape({d1, d2, d3, d4}), _array(d1 * d2 * d3 * d4) {}

template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(size_t d1, size_t d2, size_t d3, size_t d4, size_t d5): _size(d1 * d2 * d3 * d4 * d5), _shape({d1, d2, d3, d4, d5}), _array(d1 * d2 * d3 * d4 * d5) {}


template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(const std::vector<size_t> &shape): _size(1), _shape(shape), _array()
{
    for(size_t idx = 0; idx < _shape.size(); idx++)
    {
        _size *= _shape[idx]; 
    }
    _array = std::vector<Scalar>(_size); 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(const std::vector<size_t> &shape, const std::vector<Scalar> &arr): _size(arr.size()), _shape(shape), _array(arr) {}


template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(const NDArrayFloatCPU<Scalar> &ndarray): _size(ndarray._size), _shape(ndarray._shape), _array(ndarray._array) { }

template<typename Scalar>
NDArrayFloatCPU<Scalar>::NDArrayFloatCPU(NDArrayFloatCPU<Scalar> &&ndarray): _size(ndarray._size), _shape(std::move(ndarray._shape)), _array(std::move(ndarray._array)) { }


template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::init(Scalar val)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] = val; 
    }
    
    return *this; 
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::init(const std::function<Scalar()> &func)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] = func(); 
    }
    
    return *this; 
}


template<typename Scalar>
const std::vector<size_t> &NDArrayFloatCPU<Scalar>::shape() const {return _shape; }

template<typename Scalar>
size_t NDArrayFloatCPU<Scalar>::size() const {return _size; }

template<typename Scalar>
size_t NDArrayFloatCPU<Scalar>::length() const {return _size; }

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::resize(const std::vector<size_t> &shape)
{
    assert(_mulUp(shape) == _mulUp(_shape)); 
    
    _shape = shape;
    
    return *this; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::reshape(const std::vector<size_t> &shape) const
{
    assert(_mulUp(shape) == _mulUp(_shape)); 
    
    NDArrayFloatCPU<Scalar> result = *this; 
    result.resize(shape); 
    
    return result; 
}


template<typename Scalar>
Scalar &NDArrayFloatCPU<Scalar>::operator [] (size_t index)
{
    assert(index < _size); 
    
    return _array[index]; 
}

template<typename Scalar>
const Scalar &NDArrayFloatCPU<Scalar>::operator [] (size_t index) const
{
    assert(index < _size); 
    
    return _array[index]; 
}

template<typename Scalar>
Scalar &NDArrayFloatCPU<Scalar>::operator [] (const std::vector<size_t> &index)
{
    return _array[_getIndex(index)]; 
}

template<typename Scalar>
const Scalar &NDArrayFloatCPU<Scalar>::operator [] (const std::vector<size_t> &index) const
{
    return _array[_getIndex(index)]; 
} 

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::get(const std::vector<Range> &index) const
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
    NDArrayFloatCPU result(shape); 
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
void NDArrayFloatCPU<Scalar>::set(const std::vector<Range> &index, const NDArrayFloatCPU &value)
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
void NDArrayFloatCPU<Scalar>::print() const
{
    std::vector<size_t> current(_shape.size(), 0); 
    std::cout << std::endl << "NDArrayFloatCPU: " << std::endl; 
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
        std::cout << _array[_getIndex(current)] << ", "; 
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
std::vector<float> NDArrayFloatCPU<Scalar>::values() const
{
    std::vector<float> result(_shape); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result[idx] = float(_array[idx]); 
    }
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<float> NDArrayFloatCPU<Scalar>::eval() const
{
    NDArrayFloatCPU<float> result(_shape); 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result[idx] = float(_array[idx]); 
    }
    
    return result; 
}


template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator = (const NDArrayFloatCPU<Scalar> &ndarray)
{
    _size = ndarray._size; 
    _shape = ndarray._shape; 
    _array = ndarray._array; 
    
    return *this; 
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator = (NDArrayFloatCPU<Scalar> &&ndarray)
{
    _size = ndarray._size; 
    _shape = std::move(ndarray._shape); 
    _array = std::move(ndarray._array); 
    
    return *this; 
}


template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator += (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] += value; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator -= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] -= value; 
    }
    
    return *this;
} 

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator &= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] *= value; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator |= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] /= value; 
    }
    
    return *this;
} 

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator *= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] *= value; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator /= (const Scalar &value)
{
    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] /= value; 
    }
    
    return *this;
} 

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator += (const NDArrayFloatCPU<Scalar> &ndarray)
{
    assert(_mulUp(ndarray._shape) == _mulUp(_shape)); 

    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] += ndarray._array[idx]; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator -= (const NDArrayFloatCPU<Scalar> &ndarray)
{
    assert(_mulUp(ndarray._shape) == _mulUp(_shape)); 

    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] -= ndarray._array[idx]; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator &= (const NDArrayFloatCPU<Scalar> &ndarray)
{
    assert(_mulUp(ndarray._shape) == _mulUp(_shape)); 

    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] *= ndarray._array[idx]; 
    }
    
    return *this;
}

template<typename Scalar>
const NDArrayFloatCPU<Scalar> &NDArrayFloatCPU<Scalar>::operator |= (const NDArrayFloatCPU<Scalar> &ndarray)
{
    assert(_mulUp(ndarray._shape) == _mulUp(_shape)); 

    for(size_t idx = 0; idx < _size; idx++)
    {
        _array[idx] /= ndarray._array[idx]; 
    }
    
    return *this;
} 


template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator + () const {return *this; }

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator - () const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result._array[idx] = -_array[idx]; 
    }
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator + (const Scalar &value) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result += value; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator - (const Scalar &value) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result -= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator & (const Scalar &value) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result &= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator | (const Scalar &value) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result |= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator * (const Scalar &value) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result *= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator / (const Scalar &value) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result /= value; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator + (const NDArrayFloatCPU &ndarray) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result += ndarray; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator - (const NDArrayFloatCPU &ndarray) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result -= ndarray; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator & (const NDArrayFloatCPU &ndarray) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result &= ndarray; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator | (const NDArrayFloatCPU &ndarray) const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    result |= ndarray; 
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::operator * (const NDArrayFloatCPU &ndarray) const
{
    assert(_shape.size() == 2 && ndarray._shape.size() == 2);
    assert(_shape[1] == ndarray._shape[0]);
    
    size_t tmpIndex = 0, tmp1 = 0, tmp2 = 0; 
    Scalar tmpSum; 
    NDArrayFloatCPU<Scalar> tmp = ndarray.T(); 
    NDArrayFloatCPU<Scalar> result(_shape[0], tmp._shape[0]); 
    for(size_t idx = 0; idx < _shape[0]; idx++)
    {
        for(size_t jdx = 0; jdx < tmp._shape[0]; jdx++)
        {
            tmp1 = idx * _shape[1]; 
            tmp2 = jdx * tmp._shape[1]; 
            tmpSum = 0.0; 
            for(size_t kdx = 0; kdx < tmp._shape[1]; kdx++)
            {
                tmpSum += _array[tmp1++] * tmp._array[tmp2++]; 
            }
            result._array[tmpIndex++] = tmpSum; 
        }
    }
    
    return result; 
}


template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::T() const
{
    assert(_shape.size() == 2); 
    
    constexpr size_t SizeBlock = __APPROXFLOW_BLOCK_SIZE_TRANSPOSE__; 
    const size_t nBlocksX = _shape[0] / SizeBlock, nBlocksY = _shape[1] / SizeBlock; 
    size_t index1, index2, indexpre1, indexpre2; 
    
    NDArrayFloatCPU<Scalar> result(_shape[1], _shape[0]); 
    
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
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::T(const std::vector<size_t> &axis) const
{
    assert(axis.size() == _shape.size()); 
    
    std::vector<size_t> shape(_shape.size()); 
    for(size_t idx = 0; idx < _shape.size(); idx++)
    {
        shape[idx] = _shape[axis[idx]]; 
    }
    
    NDArrayFloatCPU<Scalar> result(shape); 
    
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
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::ReLU() const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    
    for(size_t idx = 0; idx < _size; idx++)
    {
        if(result._array[idx] < float(0.0))
        {
            result._array[idx] = 0.0; 
        }
    }
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::abs() const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    
    for(size_t idx = 0; idx < _size; idx++)
    {
        if(result._array[idx] < float(0.0))
        {
            result._array[idx] = 0.0; 
        }
    }
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::mask01() const
{
    NDArrayFloatCPU<Scalar> result = *this; 
    
    for(size_t idx = 0; idx < _size; idx++)
    {
        if(result._array[idx] < float(0.0))
        {
            result._array[idx] = 0.0; 
        }
        else
        {
            result._array[idx] = 1.0; 
        }
    }
    
    return result; 
}


template<typename Scalar>
std::vector<size_t> NDArrayFloatCPU<Scalar>::argmax() const
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
size_t NDArrayFloatCPU<Scalar>::posmax() const
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
Scalar NDArrayFloatCPU<Scalar>::sum() const
{
    Scalar tmpsum = 0.0; 
    for(size_t idx = 0; idx < _shape[0]; idx++)
    {
        tmpsum += _array[idx]; 
    }
    return tmpsum; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::sum(size_t axis) const
{
    assert(axis < _shape.size()); 
    
    if(_shape.size() == 1)
    {
        NDArrayFloatCPU<Scalar> tmpSum(std::vector<size_t>({1, }), std::vector<Scalar>({0.0, })); 
        for(size_t idx = 0; idx < _shape[0]; idx++)
        {
            tmpSum._array[0] += _array[idx];
        }
        return tmpSum; 
    }
    else if(_shape.size() == 2)
    {
        size_t size; 
        if(axis == 0)
        {
            size = _shape[1]; 
        }
        else
        {
            size = _shape[0]; 
        }
        NDArrayFloatCPU<Scalar> tmpSum({size}); 
        tmpSum.init(0.0); 
        if(axis == 0)
        {
            for(size_t idx = 0; idx < _shape[0]; idx++)
            {
                for(size_t jdx = 0; jdx < _shape[1]; jdx++)
                {
                    tmpSum._array[jdx] += _array[_getIndex({idx, jdx})]; 
                }
            }
        }
        else
        {
            for(size_t idx = 0; idx < _shape[0]; idx++)
            {
                for(size_t jdx = 0; jdx < _shape[1]; jdx++)
                {
                    tmpSum._array[idx] += _array[_getIndex({idx, jdx})]; 
                }
            }
        }
        return tmpSum; 
    }
    else
    {
        std::vector<size_t> shape(_shape.size() - 1); 
        size_t tmpidx = 0; 
        for(size_t idx = 0; idx < _shape.size(); idx++)
        {
            if(idx == axis)
            {
                continue; 
            }
            shape[tmpidx++] = _shape[idx]; 
        }
        
        NDArrayFloatCPU<Scalar> result(shape); 
        result.init(0.0); 
        
        std::vector<size_t> current(shape.size(), 0); 
        std::vector<size_t> tmpcurrent(_shape.size()); 
        for(size_t idx = 0; idx < result._size; idx++)
        {
            size_t tmpkdx = 0; 
            for(size_t kdx = 0; kdx < _shape.size(); kdx++)
            {
                if(kdx == axis)
                {
                    tmpcurrent[kdx] = 0; 
                }
                else
                {
                    tmpcurrent[kdx] = current[tmpkdx++]; 
                }
            }
            for(size_t jdx = 0; jdx < _shape[axis]; jdx++)
            {
                tmpcurrent[axis] = jdx; 
//                 std::cout << _getIndex(current) << ", " << _getIndex(tmpcurrent) << std::endl; 
                result._array[result._getIndex(current)] += _array[_getIndex(tmpcurrent)]; 
            }
            current[current.size() - 1] += 1; 
            for(size_t jdx = 0; jdx < shape.size(); jdx++)
            {
                if(current[current.size() - 1 - jdx] >= shape[current.size() - 1 - jdx] && current.size() - 1 - jdx > 0)
                {
                    current[current.size() - 1 - jdx] = 0;
                    current[current.size() - 2 - jdx] += 1; 
                }
            }
        }
        return result; 
    }
    //TODO Test it
    
    return NDArrayFloatCPU<Scalar>(std::vector<size_t>({1, }), std::vector<Scalar>({0.0, })) ; 
}


template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::addChWise(const NDArrayFloatCPU<Scalar> &ndarray) const
{
    assert(ndarray._shape[0] == _shape[2]); 
    assert(ndarray._shape.size() == 1); 
    assert(_shape.size() == 3); 
    
    NDArrayFloatCPU<Scalar> result = *this; 
    
    size_t index = 0; 
    for(size_t idx = 0; idx < _size; idx++)
    {
        result._array[idx] += ndarray._array[index]; 
        index = (index + 1) % ndarray._shape[0]; 
    }
    
    return result; 
}

template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::addChWiseRev(const NDArrayFloatCPU<Scalar> &ndarray) const
{
    assert(_shape[0] == ndarray._shape[2]); 
    assert(ndarray._shape.size() == 3); 
    assert(_shape.size() == 1); 
    
    NDArrayFloatCPU<Scalar> result = *this; 
    
    size_t index = 0; 
    for(size_t idx = 0; idx < ndarray._size; idx++)
    {
        result._array[index] += ndarray._array[idx]; 
        index = (index + 1) % result._shape[0]; 
    }
    
    return result; 
}


template<typename Scalar>
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::maxPool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
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
//         assert((_shape[0] - size[0] + 1) % stride[0] == 0 && (_shape[1] - size[1] + 1) % stride[1] == 0); 
        
        shape[0] = (_shape[0] - size[0] + 1 + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] - size[1] + 1 + stride[0] - 1) / stride[1]; 
    }
    
    NDArrayFloatCPU<Scalar> result(shape[0], shape[1], _shape[2]); 
    
    if(padding)
    {
        size_t tmp0 = shape[0] * stride[0] + (size[0] - stride[0]) - _shape[0]; 
		size_t tmp1 = shape[1] * stride[1] + (size[1] - stride[1]) - _shape[1]; 
        size_t padX = (tmp0 / 2) > 0 ? (tmp0 / 2) : 0;
        size_t padY = (tmp1 / 2) > 0 ? (tmp1 / 2) : 0;
        //std::cout << "Pad Size" << padX << std::endl; 
        size_t index1, index2; 
        for(size_t idx = 0; idx < shape[0]; idx++)
        {
            for(size_t jdx = 0; jdx < shape[1]; jdx++)
            {
                std::vector<size_t> maxIndex(_shape[2]); 
                std::vector<Scalar> maxVals(_shape[2], 0.0); 
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
                std::vector<Scalar> maxVals(_shape[2], 0.0); 
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
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::im2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
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
    
    NDArrayFloatCPU<Scalar> result(shape[0] * shape[1], size[0] * size[1] * _shape[2]); 
    
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
NDArrayFloatCPU<Scalar> NDArrayFloatCPU<Scalar>::col2im(const std::vector<size_t> &origin, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
{
    assert(_shape.size() == 2); 
    assert(origin.size() == 3); 
    assert(size.size() == 2); 
    assert(stride.size() == 2); 
    
    std::vector<size_t> shape(2); 
    if(padding)
    {
        shape[0] = (origin[0] + stride[0] - 1) / stride[0]; 
        shape[1] = (origin[1] + stride[1] - 1) / stride[1]; 
    }
    else
    {
        assert((origin[0] - size[0] + 1) % stride[0] == 0 && (origin[1] - size[1] + 1) % stride[1] == 0); 
        
        shape[0] = (origin[0] - size[0] + 1 + stride[0] - 1) / stride[0]; 
        shape[1] = (origin[1] - size[1] + 1 + stride[0] - 1) / stride[1]; 
    }
    
    NDArrayFloatCPU<Scalar> im(origin[0],  origin[1], origin[2]); 
    std::vector<float> timesAdded(origin[0] * origin[1] * origin[2], 0); 
    im.init(0.0); 
    
    if(padding)
    {
        size_t tmp0 = origin[0] * stride[0] + (origin[0] - stride[0]) - _shape[0]; 
		size_t tmp1 = origin[1] * stride[1] + (origin[1] - stride[1]) - _shape[1]; 
        size_t padX = tmp0 / 2, padY = tmp1 / 2; 
        size_t index1, index2; 
        for(size_t idx = 0; idx < shape[0]; idx++)
        {
            for(size_t jdx = 0; jdx < shape[1]; jdx++)
            {
                for(size_t kdx = 0; kdx < size[0]; kdx++)
                {
                    for(size_t ldx = 0; ldx < size[1]; ldx++)
                    {
                        for(size_t mdx = 0; mdx < origin[2]; mdx++)
                        {
                            if(!(idx + kdx < padX || idx + kdx >= shape[0] + padX
                                || jdx + ldx < padY || jdx + ldx >= shape[1] + padY))
                            {
                                index1 = ((idx - padX) * stride[0] * origin[1] + (jdx - padY) * stride[1]) * origin[2]
                                            + (kdx * origin[1] + ldx) * origin[2] + mdx; 
                                index2 = (idx * shape[1] + jdx) * size[0]*size[1] * origin[2]
                                            + (kdx * size[1] + ldx) * origin[2] + mdx; 
                                im._array[index1] += _array[index2]; 
                                timesAdded[index1] += 1.0; 
                            }
                        }
                    }
                }
            }
        } 
        for(size_t idx = 0; idx < im._size; idx++)
        {
            im._array[idx] *= (1.0 / timesAdded[idx]); 
        }
    }
    else
    {
        size_t begin1, begin2, index1, index2; 
        for(size_t idx = 0; idx < shape[0]; idx++)
        {
            for(size_t jdx = 0; jdx < shape[1]; jdx++)
            {
                begin1 = (idx * stride[0] * origin[1] + jdx * stride[1]) * origin[2]; 
                begin2 = (idx * shape[1] + jdx) * size[0]*size[1] * origin[2]; 
                for(size_t kdx = 0; kdx < size[0]; kdx++)
                {
                    for(size_t ldx = 0; ldx < size[1]; ldx++)
                    {
                        for(size_t mdx = 0; mdx < origin[2]; mdx++)
                        {
                            index1 = begin1 + ldx * origin[2] + mdx; 
                            index2 = begin2 + ldx * origin[2] + mdx; 
                            im._array[index1] += _array[index2]; 
                            timesAdded[index1]++; 
                        }
                    }
                    begin1 += origin[1] * origin[2]; 
                    begin2 += size[1] * origin[2]; 
                }
            }
        }
        for(size_t idx = 0; idx < im._size; idx++)
        {
            im._array[idx] *= (1.0 / timesAdded[idx]); 
        }
    }
    
    return im; 
} 

template<typename Scalar>
std::vector<size_t> NDArrayFloatCPU<Scalar>::sizeIm2col(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
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
std::vector<size_t> NDArrayFloatCPU<Scalar>::sizeIm2Col_ImOnly(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
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
std::vector<size_t> NDArrayFloatCPU<Scalar>::sizePool(const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding) const
{
    assert(_shape.size() == 3); 
    assert(size.size() == 2); 
    assert(stride.size() == 2); 
    
    std::vector<size_t> shape(3); 
    if(padding)
    {
        shape[0] = (_shape[0] + stride[0] - 1) / stride[0]; 
        shape[1] = (_shape[1] + stride[0] - 1) / stride[1]; 
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

}

#endif
