#ifndef __APPROXFLOW_OPSQUANT_NEURALNET__
#define __APPROXFLOW_OPSQUANT_NEURALNET__

#include "../Common.h"

#include "../Variable.h"
#include "../Operator.h"

namespace ApproxFlow
{

template<typename NDArrayType>
class _Conv2DReLU: public Operator<NDArrayType>
{
public:
    typedef typename NDArrayType::FloatType FloatType; 
    typedef typename NDArrayType::AddType AddType; 
    
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
    NDArrayType _col; 
    std::vector<size_t> _size; 
    std::vector<size_t> _stride; 
    bool _padding; 
    
    FloatType _S; 
    AddType _Z; 
    Ref<std::vector<typename NDArrayType::AccumType>> _biases; 
    
public:
    _Conv2DReLU(FloatType S, AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Conv2DReLU(typename NDArrayType::FloatType S, typename NDArrayType::AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true)
{
    Operator<NDArrayType> *op = new _Conv2DReLU<NDArrayType>(S, Z, x1, x2, x3, size, stride, padding); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_Conv2DReLU<NDArrayType>::_Conv2DReLU(FloatType S, AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding): Operator<NDArrayType>(), _size(size), _stride(stride), _padding(padding), _S(S), _Z(Z), _biases(x3)
{
    Operator<NDArrayType>::_type = "Conv2DReLU"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Conv2DReLU<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    std::vector<size_t> shapeTmp2 = ref1.sizeIm2Col_ImOnly(_size, _stride, _padding);  
    shapeTmp2.push_back(ref2.shape()[1]); 
    _col = ref1.im2col(_size, _stride, _padding); 
    NDArrayType tmp = NDArrayType::matMADReLU(_S, _Z, _col, ref2, *_biases); 
    tmp.resize(shapeTmp2); 
    return tmp; 
}

template<typename NDArrayType>
void _Conv2DReLU<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
    }
}

template<typename NDArrayType>
class _MaxPool: public Operator<NDArrayType>
{
public:
    typedef typename NDArrayType::FloatType FloatType; 
    typedef typename NDArrayType::AddType AddType; 
    
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
    std::vector<size_t> _size; 
    std::vector<size_t> _stride; 
    bool _padding; 
    
public:
    _MaxPool(Ref<Variable<NDArrayType>> &x1, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> MaxPool(Ref<Variable<NDArrayType>> &x1, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true)
{
    Operator<NDArrayType> *op = new _MaxPool<NDArrayType>(x1, size, stride, padding); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_MaxPool<NDArrayType>::_MaxPool(Ref<Variable<NDArrayType>> &x1, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding): Operator<NDArrayType>(), _size(size), _stride(stride), _padding(padding)
{
    Operator<NDArrayType>::_type = "MaxPool"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    x1->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _MaxPool<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    return ref1.maxPool(_size, _stride, _padding); 
}

template<typename NDArrayType>
void _MaxPool<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
    }
}

}

#endif
