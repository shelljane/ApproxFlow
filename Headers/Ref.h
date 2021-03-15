#ifndef __APPROXFLOW_REF__
#define __APPROXFLOW_REF__

#include "Common.h"

namespace ApproxFlow
{
template <typename Type>
class Ref
{
private:
    struct Container
    {
        static void _nonsense(Type&) {}
    
        std::atomic<size_t> _count; 
        Type *_pVal; 
        std::function<void(Type&)> _destroy; 
        Container(Type *pVal): _count(1), _pVal(pVal), _destroy(_nonsense) {}
        ~Container() {if(_pVal){delete _pVal;} }
    }*_container; 
    
public:
    Ref(): _container(new Container(nullptr)) {} 
    Ref(Type *pVal): _container(new Container(pVal)) {}
    Ref(const Type &val): _container(new Container(new Type(val))) {}
    Ref(Type &&val): _container(new Container(new Type(val))) {}
    Ref(const Ref<Type>& ref); 
    ~Ref(); 
    
    void onDestroy(const std::function<void(Type&)> &func) {_container->_destroy = func; }
    
    bool isEmpty() {return (_container->_pVal == nullptr); }
    const Ref<Type> &operator = (const Ref<Type>& ref); 
    Type &operator * () {return *(_container->_pVal); }
    Type &value() {return *(_container->_pVal); }
    Type *pointer() {return _container->_pVal; }
    Type *operator -> () {return _container->_pVal; }
    const Type &operator * () const {return *(_container->_pVal); }
    const Type &value() const {return *(_container->_pVal); }
    const Type *pointer() const {return _container->_pVal; }
    const Type *operator -> () const {return _container->_pVal; }
    
}; 

template <typename Type>
Ref<Type>::Ref(const Ref<Type> &ref)
{
    _container = ref._container; 
    _container->_count++; 
}
    
template <typename Type>
const Ref<Type> &Ref<Type>::operator = (const Ref<Type>& ref)
{
    if(_container != ref._container)
    {
        _container->_count--; 
        if(_container->_count == 0)
        {
            if(_container->_pVal)
            {
                _container->_destroy(*(_container->_pVal)); 
            }
            delete _container; 
        }
        _container = ref._container; 
        _container->_count++; 
    }
    else
    {
        _container->_count++; 
    }
    
    return *this; 
}

template <typename Type>
Ref<Type>::~Ref()
{
    _container->_count--; 
    if(_container->_count == 0)
    {
        if(_container->_pVal)
        {
            _container->_destroy(*(_container->_pVal)); 
        }
        delete _container; 
    }
}



}

#endif // __APPROXFLOW_REF__
