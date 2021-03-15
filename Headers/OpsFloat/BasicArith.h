#ifndef __APPROXFLOW_OPSFLOAT_BASICARITH__
#define __APPROXFLOW_OPSFLOAT_BASICARITH__

#include "../Common.h"

#include "../Variable.h"
#include "../Operator.h"

namespace ApproxFlow
{
template<typename NDArrayType>
class _Add: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _Add(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Add(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    Operator<NDArrayType> *op = new _Add<NDArrayType>(x1, x2); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}

template<typename NDArrayType>
Ref<Variable<NDArrayType>> operator + (Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    return Add(x1, x2); 
}

template<typename NDArrayType>
_Add<NDArrayType>::_Add(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2): Operator<NDArrayType>() 
{
    Operator<NDArrayType>::_type = "Add"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Add<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    return ref1 + ref2; 
}

template<typename NDArrayType>
void _Add<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            Operator<NDArrayType>::_vars[0]->addGradient(Operator<NDArrayType>::_opvar->gradient()); 
            Operator<NDArrayType>::_vars[1]->addGradient(Operator<NDArrayType>::_opvar->gradient()); 
        }
    }
}


template<typename NDArrayType>
class _Subtract: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _Subtract(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Subtract(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    Operator<NDArrayType> *op = new _Subtract<NDArrayType>(x1, x2); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}

template<typename NDArrayType>
Ref<Variable<NDArrayType>> operator - (Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    return Subtract(x1, x2); 
}


template<typename NDArrayType>
_Subtract<NDArrayType>::_Subtract(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2): Operator<NDArrayType>() 
{
    Operator<NDArrayType>::_type = "Subtract"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Subtract<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    return ref1 - ref2; 
}

template<typename NDArrayType>
void _Subtract<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            Operator<NDArrayType>::_vars[0]->addGradient(Operator<NDArrayType>::_opvar->gradient()); 
            Operator<NDArrayType>::_vars[1]->addGradient(-Operator<NDArrayType>::_opvar->gradient()); 
        }
    }
}


template<typename NDArrayType>
class _Multiply: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _Multiply(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Multiply(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    Operator<NDArrayType> *op = new _Multiply<NDArrayType>(x1, x2); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}

template<typename NDArrayType>
Ref<Variable<NDArrayType>> operator & (Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    return Multiply(x1, x2); 
}


template<typename NDArrayType>
_Multiply<NDArrayType>::_Multiply(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2): Operator<NDArrayType>() 
{
    Operator<NDArrayType>::_type = "Multiply"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Multiply<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    return (ref1 & ref2); 
}

template<typename NDArrayType>
void _Multiply<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
            Operator<NDArrayType>::_vars[0]->addGradient(Operator<NDArrayType>::_opvar->gradient() & ref2); 
            Operator<NDArrayType>::_vars[1]->addGradient(Operator<NDArrayType>::_opvar->gradient() & ref1); 
        }
    }
}


template<typename NDArrayType>
class _MatMul: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _MatMul(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> MatMul(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    Operator<NDArrayType> *op = new _MatMul<NDArrayType>(x1, x2); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}

template<typename NDArrayType>
Ref<Variable<NDArrayType>> operator * (Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    return MatMul(x1, x2); 
}


template<typename NDArrayType>
_MatMul<NDArrayType>::_MatMul(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2): Operator<NDArrayType>() 
{
    Operator<NDArrayType>::_type = "MatMul"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _MatMul<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    return (ref1 * ref2); 
}

template<typename NDArrayType>
void _MatMul<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
            Operator<NDArrayType>::_vars[0]->addGradient(Operator<NDArrayType>::_opvar->gradient() * ref2.T()); 
            Operator<NDArrayType>::_vars[1]->addGradient(ref1.T() * Operator<NDArrayType>::_opvar->gradient()); 
        }
    }
}


}

#endif
