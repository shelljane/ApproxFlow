#ifndef __APPROXFLOW_OPSQUANT__
#define __APPROXFLOW_OPSQUANT__

#include "Common.h"

#include "Variable.h"
#include "Operator.h"

#include "./OpsQuant/BasicArith.h"
#include "./OpsQuant/NeuralNet.h"

namespace ApproxFlow
{

template<typename NDArrayType>
class _Flatten: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _Flatten(Ref<Variable<NDArrayType>> &x1); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Flatten(Ref<Variable<NDArrayType>> &x1)
{
    Operator<NDArrayType> *op = new _Flatten<NDArrayType>(x1); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_Flatten<NDArrayType>::_Flatten(Ref<Variable<NDArrayType>> &x1): Operator<NDArrayType>()
{
    Operator<NDArrayType>::_type = "FlattenAll"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    x1->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Flatten<NDArrayType>::evaluate(size_t step)
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
void _Flatten<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
    }
}

}


#endif
