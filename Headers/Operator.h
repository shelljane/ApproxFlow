#ifndef __APPROXFLOW_OPERATOR__
#define __APPROXFLOW_OPERATOR__

#include "Common.h"

#include "Range.h"
#include "Ref.h"

namespace ApproxFlow
{
template<typename NDArrayType>
class Variable; 
    
template<typename NDArrayType>
class Operator
{
protected: 
    std::string _type; 
    size_t _step, _stepdiff; 
    
    Variable<NDArrayType> *_opvar; 
    std::vector<Ref<Variable<NDArrayType>>> _vars; 
    
public:
    Operator(): _type("Operator"), _step(0), _stepdiff(0), _opvar(nullptr), _vars() {}
    virtual ~Operator() {}
    
    const std::string &type() const {return _type; }
    size_t step() const {return _step; }
    size_t stepdiff() const  {return _stepdiff; } 
    
    Variable<NDArrayType> *opvar() {return _opvar; } 
    Variable<NDArrayType> *opvar(Variable<NDArrayType> *opvar) {return (_opvar = opvar); } 
    std::vector<Ref<Variable<NDArrayType>>> &vars() {return _vars; } 
    const std::vector<Ref<Variable<NDArrayType>>> &vars() const {return _vars; } 
    
    virtual NDArrayType evaluate(size_t step) = 0; 
    virtual void differentiate(size_t step) = 0; 
}; 
}

#endif

