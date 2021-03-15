#ifndef __APPROXFLOW_OPSFLOAT__
#define __APPROXFLOW_OPSFLOAT__

#include "Common.h"

#include "Variable.h"
#include "Operator.h"

#include "./OpsFloat/BasicArith.h"
#include "./OpsFloat/NeuralNet.h"

namespace ApproxFlow
{

template<typename NDArrayType>
class _AddBias: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _AddBias(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> AddBias(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    Operator<NDArrayType> *op = new _AddBias<NDArrayType>(x1, x2); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_AddBias<NDArrayType>::_AddBias(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2): Operator<NDArrayType>() 
{
    Operator<NDArrayType>::_type = "AddBias"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _AddBias<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    return ref1.addChWise(ref2); 
}

template<typename NDArrayType>
void _AddBias<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &grad = Operator<NDArrayType>::_opvar->gradient(); 
            NDArrayType tmpgrad(grad.shape()[2]); 
            tmpgrad.init(0.0); 
            Operator<NDArrayType>::_vars[0]->addGradient(grad); 
            Operator<NDArrayType>::_vars[1]->addGradient(tmpgrad.addChWiseRev(grad)); 
        }
    }
}







template<typename NDArrayType>
class _Mean: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
    size_t _axis; 
public:
    _Mean(Ref<Variable<NDArrayType>> &x1, size_t axis=0); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Mean(Ref<Variable<NDArrayType>> &x1, size_t axis)
{
    Operator<NDArrayType> *op = new _Mean<NDArrayType>(x1, axis); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_Mean<NDArrayType>::_Mean(Ref<Variable<NDArrayType>> &x1, size_t axis): Operator<NDArrayType>(), _axis(axis)
{
    Operator<NDArrayType>::_type = "Mean"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    x1->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Mean<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    return (ref1.sum(_axis) | ref1.shape()[_axis]); 
}

template<typename NDArrayType>
void _Mean<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            NDArrayType tmp(ref1.shape()); 
            tmp.init(1.0 / ref1.shape()[_axis]); 
            Operator<NDArrayType>::_vars[0]->addGradient(tmp); 
        }
    }
}


template<typename NDArrayType>
class _FlattenAll: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _FlattenAll(Ref<Variable<NDArrayType>> &x1); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> FlattenAll(Ref<Variable<NDArrayType>> &x1)
{
    Operator<NDArrayType> *op = new _FlattenAll<NDArrayType>(x1); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_FlattenAll<NDArrayType>::_FlattenAll(Ref<Variable<NDArrayType>> &x1): Operator<NDArrayType>()
{
    Operator<NDArrayType>::_type = "FlattenAll"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    x1->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _FlattenAll<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    return ref1.reshape(std::vector<size_t>({1, ref1.size()})); 
}

template<typename NDArrayType>
void _FlattenAll<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            const NDArrayType &ref2 = Operator<NDArrayType>::_opvar->gradient(); 
            Operator<NDArrayType>::_vars[0]->addGradient(ref2.reshape(ref1.shape())); 
        }
    }
}




template<typename NDArrayType>
class _Im2Col: public Operator<NDArrayType>
{
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
    _Im2Col(Ref<Variable<NDArrayType>> &x1, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Im2Col(Ref<Variable<NDArrayType>> &x1, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true)
{
    Operator<NDArrayType> *op = new _Im2Col<NDArrayType>(x1, size, stride, padding); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_Im2Col<NDArrayType>::_Im2Col(Ref<Variable<NDArrayType>> &x1, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding): Operator<NDArrayType>(), _size(size), _stride(stride), _padding(padding)
{
    Operator<NDArrayType>::_type = "Im2Col"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    x1->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Im2Col<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    return ref1.im2col(_size, _stride, _padding); 
}

template<typename NDArrayType>
void _Im2Col<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            const NDArrayType &ref2 = Operator<NDArrayType>::_opvar->gradient(); 
            Operator<NDArrayType>::_vars[0]->addGradient(ref2.col2im(ref1.shape(), _size, _stride, _padding)); 
        }
    }
}




}

#endif
