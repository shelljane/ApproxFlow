#ifndef __APPROXFLOW_OPSQUANT_BASICARITH__
#define __APPROXFLOW_OPSQUANT_BASICARITH__



namespace ApproxFlow
{
    
template<typename NDArrayType>
class _MAD: public Operator<NDArrayType>
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
    FloatType _S; 
    AddType _Z; 
    Ref<std::vector<typename NDArrayType::AccumType>> _biases; 
    
public:
    
    _MAD(FloatType S, AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> MAD(typename _MAD<NDArrayType>::FloatType S, typename _MAD<NDArrayType>::AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3)
{
    Operator<NDArrayType> *op = new _MAD<NDArrayType>(S, Z, x1, x2, x3); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}

template<typename NDArrayType>
_MAD<NDArrayType>::_MAD(FloatType S, AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3): Operator<NDArrayType>(), _S(S), _Z(Z), _biases(x3)
{
    Operator<NDArrayType>::_type = "MAD"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _MAD<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    return NDArrayType::matMAD(_S, _Z, ref1, ref2, *_biases); 
}

template<typename NDArrayType>
void _MAD<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
    }
}




template<typename NDArrayType>
class _MADReLU: public Operator<NDArrayType>
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
    FloatType _S; 
    AddType _Z; 
    Ref<std::vector<typename NDArrayType::AccumType>> _biases; 
    
public:
    
    _MADReLU(FloatType S, AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3); 
    
    virtual NDArrayType evaluate(size_t step); 
    virtual void differentiate(size_t step); 
};

template<typename NDArrayType>
Ref<Variable<NDArrayType>> MADReLU(typename _MADReLU<NDArrayType>::FloatType S, typename _MADReLU<NDArrayType>::AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3)
{
    Operator<NDArrayType> *op = new _MADReLU<NDArrayType>(S, Z, x1, x2, x3); 
    Variable<NDArrayType> *var = new Variable<NDArrayType>(op); 
    op->opvar(var); 
    return Ref<Variable<NDArrayType>>(var); 
}

template<typename NDArrayType>
_MADReLU<NDArrayType>::_MADReLU(FloatType S, AddType Z, Ref<Variable<NDArrayType>> &x1, Ref<Variable<NDArrayType>> &x2, Ref<std::vector<typename NDArrayType::AccumType>> &x3): Operator<NDArrayType>(), _S(S), _Z(Z), _biases(x3)
{
    Operator<NDArrayType>::_type = "MADReLU"; 
    Operator<NDArrayType>::_vars.push_back(x1); 
    Operator<NDArrayType>::_vars.push_back(x2); 
    x1->usedBy(this); 
    x2->usedBy(this); 
}

template<typename NDArrayType>
NDArrayType _MADReLU<NDArrayType>::evaluate(size_t step)
{
    if(Operator<NDArrayType>::_step < step)
    {
        Operator<NDArrayType>::_step = step; 
    }
    Operator<NDArrayType>::_vars[0]->evaluate(step); 
    Operator<NDArrayType>::_vars[1]->evaluate(step); 
    const NDArrayType &ref1 = Operator<NDArrayType>::_vars[0]->value(); 
    const NDArrayType &ref2 = Operator<NDArrayType>::_vars[1]->value(); 
    return NDArrayType::matMADReLU(_S, _Z, ref1, ref2, *_biases); 
}

template<typename NDArrayType>
void _MADReLU<NDArrayType>::differentiate(size_t step)
{
    if(Operator<NDArrayType>::_stepdiff < step)
    {
        Operator<NDArrayType>::_stepdiff = step; 
    }
}

}


#endif
