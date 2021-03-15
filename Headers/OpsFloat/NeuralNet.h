#ifndef __APPROXFLOW_OPSFLOAT_NEURALNET__
#define __APPROXFLOW_OPSFLOAT_NEURALNET__

#include "../Common.h"

#include "../Variable.h"
#include "../Operator.h"

namespace ApproxFlow
{
    

template<typename NDArrayType>
class _MSE_Loss: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _MSE_Loss(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> MSE_Loss(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2)
{
    Operator<NDArrayType> *op = new _MSE_Loss<NDArrayType>(x1, x2); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_MSE_Loss<NDArrayType>::_MSE_Loss(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2): Operator<NDArrayType>() 
{
    Operator<NDArrayType>::_type = "MSE_Loss"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _MSE_Loss<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    NDArrayType tmp = (ref2 - ref1); 
    tmp = (tmp & tmp).sum(1); 
    return tmp.sum(0) / tmp.shape()[0]; 
}

template<typename NDArrayType>
void _MSE_Loss<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
            NDArrayType grad = (ref2 - ref1) * 2.0; 
            Operator<NDArrayType>::_vars[0]->addGradient(-grad * (1.0 / grad.shape()[0])); 
            Operator<NDArrayType>::_vars[1]->addGradient(grad * (1.0 / grad.shape()[0])); 
        }
    }
}


template<typename NDArrayType>
class _ReLU: public Operator<NDArrayType>
{
protected: 
/*
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
*/
public:
    _ReLU(Ref<Variable<NDArrayType>> &x1); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> ReLU(Ref<Variable<NDArrayType>> &x1)
{
    Operator<NDArrayType> *op = new _ReLU<NDArrayType>(x1); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_ReLU<NDArrayType>::_ReLU(Ref<Variable<NDArrayType>> &x1): Operator<NDArrayType>()
{
    Operator<NDArrayType>::_type = "ReLU"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    x1->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _ReLU<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    return ref1.ReLU(); 
}

template<typename NDArrayType>
void _ReLU<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            const NDArrayType &ref2 = Operator<NDArrayType>::_opvar->gradient(); 
            Operator<NDArrayType>::_vars[0]->addGradient(ref2 & ref1.mask01()); 
        }
    }
}


template<typename NDArrayType>
class _Conv2D: public Operator<NDArrayType>
{
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
public:
    _Conv2D(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> Conv2D(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding=true)
{
    Operator<NDArrayType> *op = new _Conv2D<NDArrayType>(x1, x2, size, stride, padding); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}


template<typename NDArrayType>
_Conv2D<NDArrayType>::_Conv2D(Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, const std::vector<size_t> &size, const std::vector<size_t> &stride, bool padding): Operator<NDArrayType>(), _size(size), _stride(stride), _padding(padding)
{
    Operator<NDArrayType>::_type = "Conv2D"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _Conv2D<NDArrayType>::evaluate(size_t step)
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
    NDArrayType tmp = _col * ref2; 
    tmp.resize(shapeTmp2); 
    return tmp; 
}

template<typename NDArrayType>
void _Conv2D<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
        if(Operator<NDArrayType>::_opvar != nullptr)
        {
            Operator<NDArrayType>::_opvar->differentiate(step); 
            const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
            const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
            NDArrayType grad = Operator<NDArrayType>::_opvar->gradient(); 
            print(grad.shape()); 
            grad.resize(std::vector<size_t>({grad.shape()[0]*grad.shape()[1], grad.shape()[2]})); 
            
            NDArrayType gradIm = grad * ref2.T(); 
            gradIm = gradIm.col2im(ref1.shape(), _size, _stride, _padding); 
            
            Operator<NDArrayType>::_vars[0]->addGradient(gradIm); 
            Operator<NDArrayType>::_vars[1]->addGradient(_col.T() * grad); 
        }
    }
}
    
}

#endif
