#ifndef __APPROXFLOW_VARIABLE__
#define __APPROXFLOW_VARIABLE__

#include "Common.h"

#include "Range.h"
#include "Ref.h"

#include "Operator.h"

namespace ApproxFlow
{

template <typename NDArrayType>
class Variable
{
private: 
    size_t _step, _stepdiff; 
    
    Ref<Operator<NDArrayType>> _op; 
    Ref<NDArrayType> _value; 
    Ref<NDArrayType> _gradient; 
    
    std::vector<Operator<NDArrayType> *> _usedBy; 
    
public: 
    Variable(); 
    Variable(const NDArrayType &value); 
    Variable(const Ref<NDArrayType> &value); 
    Variable(const Ref<Operator<NDArrayType>> &op); 
    Variable(const Variable<NDArrayType> &var); 
    const Variable<NDArrayType> &operator = (const Variable<NDArrayType> &var); 
    
    const NDArrayType &value() const {return *_value; }
    const NDArrayType &gradient() const {return *_gradient; }
    
    void evaluate(size_t step); 
    void differentiate(size_t step); 
    void forward(size_t step) {evaluate(step); }
    void backprop(size_t step) {differentiate(step); }
    void update() {*_value -= *_gradient; }
    
    void setValue(const NDArrayType &value) {*_value = value; }
    void setGradient(const NDArrayType &grad) {*_gradient = grad; }
    void addGradient(const NDArrayType &grad); 
    void usedBy(Operator<NDArrayType> *op) {_usedBy.push_back(op); }
}; 

template<typename NDArrayType>
Variable<NDArrayType>::Variable(): _step(0), _stepdiff(0), _op(), _value(new NDArrayType()), _gradient(new NDArrayType()) {}

template<typename NDArrayType>
Variable<NDArrayType>::Variable(const NDArrayType &value): _step(0), _stepdiff(0), _op(), _value(new NDArrayType(value)), _gradient(new NDArrayType()) {}

template<typename NDArrayType>
Variable<NDArrayType>::Variable(const Ref<NDArrayType> &value): _step(0), _stepdiff(0), _op(), _value(value), _gradient(new NDArrayType()) {}

template<typename NDArrayType>
Variable<NDArrayType>::Variable(const Ref<Operator<NDArrayType>> &op): _step(0), _stepdiff(0), _op(op), _value(new NDArrayType()), _gradient(new NDArrayType()) {}

template<typename NDArrayType>
Variable<NDArrayType>::Variable(const Variable<NDArrayType> &var): _step(var._step), _stepdiff(var._stepdiff), _op(var._op), _value(var._value), _gradient(new NDArrayType(var._gradient)) {}


template<typename NDArrayType>
const Variable<NDArrayType> &Variable<NDArrayType>::operator = (const Variable<NDArrayType> &var) 
{
    _step = var._step; 
    _stepdiff = var._stepdiff; 
    _op = var._op; 
    _value = var._value; 
}


template<typename NDArrayType>
void Variable<NDArrayType>::evaluate(size_t step)
{
    if(_step < step)
    {
        _step = step; 
        *_gradient = NDArrayType(); 
        if(!_op.isEmpty())
        {
            *_value = _op->evaluate(step); 
        }
    }
}

template<typename NDArrayType>
void Variable<NDArrayType>::differentiate(size_t step)
{
    if(_stepdiff < step)
    {
        _stepdiff = step; 
        for(size_t idx = 0; idx < _usedBy.size(); idx++)
        {
            _usedBy[idx]->differentiate(step); 
        }
    }
}


template<typename NDArrayType>
void Variable<NDArrayType>::addGradient(const NDArrayType &grad)
{
    if(_gradient->size() == 0)
    {
        *_gradient = grad; 
    }
    else
    {
        *_gradient += grad; 
    }
}
}

#endif
