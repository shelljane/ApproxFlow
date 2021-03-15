#ifndef __APPROXFLOW_THREAD__
#define __APPROXFLOW_THREAD__

#include "Common.h"

#include "Ref.h"

namespace ApproxFlow
{

class Thread
{
private:
    
    Ref<std::thread> _thread;
    
public:
    
    static void destroy(std::thread &thread)
    {
        if(thread.joinable())
        {
            thread.join();
        }
    }
    
    Thread(std::thread *t): _thread(t) { _thread.onDestroy(destroy); }
    
    void join() 
    {
        if(_thread->joinable())
        {
            _thread->join();
        }
    }
    void detach() {_thread->detach(); }
    
};

}

#endif // __APPROXFLOW_THREAD__

