#ifndef __APPROXFLOW_UTILS__
#define __APPROXFLOW_UTILS__

#include "../Headers/Common.h"
#include "../Headers/Timer.h"
#include "../Headers/Range.h"
#include "../Headers/Ref.h"
#include "../Headers/Thread.h"
#include "../Headers/NDArrayQuantCPU.h"
#include "../Headers/NDArrayApproxCPU.h"

#include "../Headers/OpsQuant.h"

using namespace ApproxFlow; 
using namespace std; 

typedef unsigned char Scalar; 
typedef unsigned short AddType; 
typedef int AccumType; 
typedef float FloatType; 
// typedef NDArrayQuantCPU<Scalar> NDArray; 
typedef NDArrayApproxCPU<Scalar> NDArray; 
typedef Variable<NDArray> Var; 

FloatType getS_0(const vector<FloatType> &arr)
{
    FloatType minNum = *min_element(arr.begin(), arr.end()); 
    FloatType maxNum = *max_element(arr.begin(), arr.end()); 
    
//     cout << "Max: " << maxNum << "; Min: " << minNum << endl; 
//     cout << "Steps: " << ((1 << (sizeof(Scalar) << 3)) - 1) << endl; 
    
    return (maxNum - minNum) / ((1 << (sizeof(Scalar) << 3)) - 1); 
}

AddType getZ_0(const vector<FloatType> &arr, FloatType S)
{
    FloatType minNum = *min_element(arr.begin(), arr.end()); 
    
    return round((0.0 - minNum) / S); 
}

vector<Scalar> getQ_0(const vector<FloatType> &arr, FloatType S, AddType Z)
{
    vector<Scalar> result(arr.size()); 
    for(size_t idx = 0; idx < arr.size(); idx++)
    {
        result[idx] = (Z + arr[idx] / S > 0) ? round(Z + arr[idx] / S) : 0; 
    }
    
    return result; 
}

vector<AccumType> getQ_0_AT(const vector<FloatType> &arr, FloatType S, AddType Z)
{
    vector<AccumType> result(arr.size()); 
    for(size_t idx = 0; idx < arr.size(); idx++)
    {
        result[idx] = round(Z + arr[idx] / S); 
    }
    
    return result; 
}

NDArray getNDArray_0(const vector<FloatType> &vec, const vector<size_t> &shape)
{
    FloatType s = getS_0(vec); 
    AddType z = getZ_0(vec, s); 
    vector<Scalar> vec_q = getQ_0(vec, s, z); 
    
    return NDArray(s, z, shape, vec_q); 
}

vector<vector<float>> getImages()
{
    vector<vector<float>> data; 
    ifstream fin("../Data/MNIST_TestData.dat"); 
    if(!fin)
    {
        cerr << "ERROR when reading files. " << endl; 
        exit(1); 
    }
    for(unsigned idx = 0; idx < 10000; idx++)
    {
        vector<float> tmpVec(784); 
        for(unsigned jdx = 0; jdx < 784; jdx++)
        {
            float tmp; 
            fin >> tmp; 
            tmpVec[jdx] = tmp; 
        }
        data.push_back(tmpVec); 
    }
    
    fin.close(); 
    
    return data; 
}

vector<unsigned> getLabels()
{
    vector<unsigned> data(10000); 
    ifstream fin("../Data/MNIST_TestLabel.dat"); 
    if(!fin)
    {
        cerr << "ERROR when reading files. " << endl; 
        exit(1); 
    }
    
    for(unsigned idx = 0; idx < 10000; idx++)
    {
        fin >> data[idx]; 
    }
    
    fin.close(); 
    
    return data; 
}

vector<vector<float>> getImagesFashion()
{
    vector<vector<float>> data; 
    ifstream fin("../Data/FashionMNIST_TestData.dat"); 
    if(!fin)
    {
        cerr << "ERROR when reading files. " << endl; 
        exit(1); 
    }
    for(unsigned idx = 0; idx < 10000; idx++)
    {
        vector<float> tmpVec(784); 
        for(unsigned jdx = 0; jdx < 784; jdx++)
        {
            float tmp; 
            fin >> tmp; 
            tmpVec[jdx] = tmp; 
        }
        data.push_back(tmpVec); 
    }
    
    fin.close(); 
    
    return data; 
}

vector<unsigned> getLabelsFashion()
{
    vector<unsigned> data(10000); 
    ifstream fin("../Data/FashionMNIST_TestLabel.dat"); 
    if(!fin)
    {
        cerr << "ERROR when reading files. " << endl; 
        exit(1); 
    }
    
    for(unsigned idx = 0; idx < 10000; idx++)
    {
        fin >> data[idx]; 
    }
    
    fin.close(); 
    
    return data; 
}

vector<vector<float>> getImagesCIFAR10()
{
    vector<vector<float>> data; 
    ifstream fin("../Data/CIFAR10_TestData.dat"); 
    if(!fin)
    {
        cerr << "ERROR when reading files. " << endl; 
        exit(1); 
    }
    for(unsigned idx = 0; idx < 10000; idx++)
    {
        vector<float> tmpVec(784*3); 
        for(unsigned jdx = 0; jdx < 784*3; jdx++)
        {
            float tmp; 
            fin >> tmp; 
            tmpVec[jdx] = tmp; 
        }
        data.push_back(tmpVec); 
    }
    
    fin.close(); 
    
    return data; 
}

vector<unsigned> getLabelsCIFAR10()
{
    vector<unsigned> data(10000); 
    ifstream fin("../Data/CIFAR10_TestLabel.dat"); 
    if(!fin)
    {
        cerr << "ERROR when reading files. " << endl; 
        exit(1); 
    }
    
    for(unsigned idx = 0; idx < 10000; idx++)
    {
        fin >> data[idx]; 
    }
    
    fin.close(); 
    
    return data; 
}


#endif
