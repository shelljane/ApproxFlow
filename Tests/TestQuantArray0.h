#ifndef __TESTQUANTARRAY0__
#define __TESTQUANTARRAY0__

#include "../Headers/Common.h"
#include "../Headers/Timer.h"
#include "../Headers/Range.h"
#include "../Headers/Ref.h"
#include "../Headers/Thread.h"
#include "../Headers/NDArrayQuantCPU.h"
#include "../Headers/NDArrayApproxCPU.h"

#include "../Headers/OpsQuant.h"

#include "../Tests/Utils.h"

int testMNIST(const string lutfile = "../Utils/LUT_HEAM.txt")
{   
    float S_input = 1.0 / 255.0; 
    int Z_input   = 0; 
    
    const size_t NUMTHREADS = 20; 
    const string WEIGHTFOLDER = "../Weights/MNIST/"; 
    
    NDArray::_loadLUT(lutfile); 
    
    unordered_map<string, float> map_S_weights; 
    unordered_map<string, int> map_Z_weights; 
    unordered_map<string, vector<Scalar>> map_Q_weights; 
    unordered_map<string, NDArray> map_weights; 
    
    unordered_map<string, float> map_S_biases; 
    unordered_map<string, int> map_Z_biases; 
    unordered_map<string, vector<int>> map_Q_biases; 
    
    unordered_map<string, float> map_S_activations; 
    unordered_map<string, int> map_Z_activations; 
    
    vector<string> namesLayers  = {"Conv1", "Conv2", "Conv3", "FC1", "FC_Logits"}; 
    vector<size_t> sizesWeights = {5*5*1*16, 5*5*16*32, 5*5*32*64, 1024*256, 256*10}; 
    vector<size_t> sizesBiases = {16, 32, 64, 256, 10}; 
    vector<vector<size_t>> shapesWeights = {{5*5*1, 16}, {5*5*16, 32}, {5*5*32, 64}, {1024, 256}, {256, 10}}; 
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_weights; 
        int Z_weights; 
        vector<Scalar> Q_weights(sizesWeights[idx]); 
        
        ifstream fin_weights(WEIGHTFOLDER + namesLayers[idx] + "_weights.txt"); 
        if(!fin_weights)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_weights >> S_weights; 
        fin_weights >> Z_weights;
        for(size_t jdx = 0; jdx < sizesWeights[idx]; jdx++)
        {
            int tmp; 
            fin_weights >> tmp; 
            Q_weights[jdx] = tmp; 
        }
        fin_weights.close(); 

        NDArray weights(S_weights, Z_weights, shapesWeights[idx], Q_weights);  
        
        map_S_weights[namesLayers[idx]] = S_weights; 
        map_Z_weights[namesLayers[idx]] = Z_weights; 
        map_Q_weights[namesLayers[idx]] = Q_weights; 
        map_weights[namesLayers[idx]]   = weights; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_biases; 
        int Z_biases; 
        vector<int> Q_biases(sizesBiases[idx]); 
        
        ifstream fin_biases(WEIGHTFOLDER + namesLayers[idx] + "_biases.txt"); 
        if(!fin_biases)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_biases >> S_biases; 
        fin_biases >> Z_biases;
        for(size_t jdx = 0; jdx < sizesBiases[idx]; jdx++)
        {
            int tmp; 
            fin_biases >> tmp; 
            Q_biases[jdx] = tmp; 
        }
        fin_biases.close(); 
        
        map_S_biases[namesLayers[idx]] = S_biases; 
        map_Z_biases[namesLayers[idx]] = Z_biases; 
        map_Q_biases[namesLayers[idx]] = Q_biases; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_activations; 
        int Z_activations; 
        
        ifstream fin_activations(WEIGHTFOLDER + namesLayers[idx] + "_activations.txt"); 
        if(!fin_activations)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_activations >> S_activations; 
        fin_activations >> Z_activations;
        fin_activations.close(); 
        
        map_S_activations[namesLayers[idx]] = S_activations; 
        map_Z_activations[namesLayers[idx]] = Z_activations; 
    }
    vector<Ref<NDArray>> image_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        image_pool.push_back(new NDArray(S_input, Z_input, {28, 28, 1})); 
    }
    vector<Ref<Var>> input_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        input_pool.push_back(new Var(image_pool[idx])); 
    }
    
    cout << "Building Conv1: S = " << map_S_weights["Conv1"] << "; Z = " << map_Z_weights["Conv1"] << "; S_act = " << map_S_activations["Conv1"] << "; Z_act = " << map_Z_activations["Conv1"] << endl; 
    
    vector<Ref<Var>> weightsConv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv1_pool.push_back(new Var(map_weights["Conv1"])); 
    }
    vector<Ref<vector<int>>> biasesConv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv1_pool.push_back(new vector<int>(map_Q_biases["Conv1"])); 
    }
    vector<Ref<Var>> preconv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv1_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv1"], map_Z_activations["Conv1"], input_pool[idx], weightsConv1_pool[idx], biasesConv1_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv1_pool.push_back(MaxPool<NDArray>(preconv1_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv2: S = " << map_S_weights["Conv2"] << "; Z = " << map_Z_weights["Conv2"] << "; S_act = " << map_S_activations["Conv2"] << "; Z_act = " << map_Z_activations["Conv2"] << endl; 
    
    vector<Ref<Var>> weightsConv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv2_pool.push_back(new Var(map_weights["Conv2"])); 
    }
    vector<Ref<vector<int>>> biasesConv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv2_pool.push_back(new vector<int>(map_Q_biases["Conv2"])); 
    }
    vector<Ref<Var>> preconv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv2_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv2"], map_Z_activations["Conv2"], conv1_pool[idx], weightsConv2_pool[idx], biasesConv2_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv2_pool.push_back(MaxPool<NDArray>(preconv2_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv3: S = " << map_S_weights["Conv3"] << "; Z = " << map_Z_weights["Conv3"] << "; S_act = " << map_S_activations["Conv3"] << "; Z_act = " << map_Z_activations["Conv3"] << endl; 
    
    vector<Ref<Var>> weightsConv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv3_pool.push_back(new Var(map_weights["Conv3"])); 
    }
    vector<Ref<vector<int>>> biasesConv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv3_pool.push_back(new vector<int>(map_Q_biases["Conv3"])); 
    }
    vector<Ref<Var>> preconv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv3_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv3"], map_Z_activations["Conv3"], conv2_pool[idx], weightsConv3_pool[idx], biasesConv3_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv3_pool.push_back(MaxPool<NDArray>(preconv3_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    vector<Ref<Var>> flatten_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        flatten_pool.push_back(Flatten<NDArray>(conv3_pool[idx])); 
    }
    
    cout << "Building FC1: S = " << map_S_weights["FC1"] << "; Z = " << map_Z_weights["FC1"] << "; S_act = " << map_S_activations["FC1"] << "; Z_act = " << map_Z_activations["FC1"] << endl; 
    
    vector<Ref<Var>> weightsFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC1_pool.push_back(new Var(map_weights["FC1"])); 
    }
    vector<Ref<vector<int>>> biasesFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC1_pool.push_back(new vector<int>(map_Q_biases["FC1"])); 
    }
    vector<Ref<Var>> fc1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc1_pool.push_back(MADReLU<NDArray>(map_S_activations["FC1"], map_Z_activations["FC1"], flatten_pool[idx], weightsFC1_pool[idx], biasesFC1_pool[idx])); 
    }
    
    cout << "Building FC2: S = " << map_S_weights["FC_Logits"] << "; Z = " << map_Z_weights["FC_Logits"] << "; S_act = " << map_S_activations["FC_Logits"] << "; Z_act = " << map_Z_activations["FC_Logits"] << endl; 
    
    vector<Ref<Var>> weightsFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC2_pool.push_back(new Var(map_weights["FC_Logits"])); 
    }
    vector<Ref<vector<int>>> biasesFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC2_pool.push_back(new vector<int>(map_Q_biases["FC_Logits"])); 
    }
    vector<Ref<Var>> fc2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc2_pool.push_back(MAD<NDArray>(map_S_activations["FC_Logits"], map_Z_activations["FC_Logits"], fc1_pool[idx], weightsFC2_pool[idx], biasesFC2_pool[idx])); 
    }
    
    vector<vector<float>> images = getImages(); 
    vector<unsigned> labels = getLabels(); 
    
    float count = 0; 
    size_t TestSize = 10000;  
    for(size_t idx = 0; idx < TestSize; idx+=NUMTHREADS)
    {
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            vector<Scalar> Q_image(784); 
            for(size_t kdx = 0; kdx < 784; kdx++)
            {
                float tmp = images[idx+jdx][kdx]; 
                Q_image[kdx] = (Z_input + tmp / S_input > 0) ? round(Z_input + tmp / S_input) : 0;
            }
            image_pool[jdx]->set(Q_image); 
        }
        
        #pragma omp parallel for num_threads(NUMTHREADS)
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            fc2_pool[jdx]->evaluate(idx+1); 
        }
        
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            size_t label = fc2_pool[jdx]->value().posmax(); 
            if(label == labels[idx+jdx])
            {
                count++; 
            }
            //fc2_pool[jdx]->value().eval().print(); 
            cout << "Sample No." << (idx+jdx+1) << " " << "; Label: " << labels[idx+jdx] << ", Predicted: " << label << " -> " << ((label == labels[idx+jdx]) ? "Right" : "Wrong") << endl; 
        }
    }
    cout << "Accuray: " << (count / TestSize) << endl; 
    
    return 0; 
}

int testFashionMNIST(const string &lutfile = "../Utils/LUT_HEAM.txt")
{   
    float S_input = 1.0 / 255.0; 
    int Z_input   = 0; 
    
    const size_t NUMTHREADS = 20; 
    const string WEIGHTFOLDER = "../Weights/FashionMNIST/"; 
    
    NDArray::_loadLUT(lutfile); 
    
    unordered_map<string, float> map_S_weights; 
    unordered_map<string, int> map_Z_weights; 
    unordered_map<string, vector<Scalar>> map_Q_weights; 
    unordered_map<string, NDArray> map_weights; 
    
    unordered_map<string, float> map_S_biases; 
    unordered_map<string, int> map_Z_biases; 
    unordered_map<string, vector<int>> map_Q_biases; 
    
    unordered_map<string, float> map_S_activations; 
    unordered_map<string, int> map_Z_activations; 
    
    vector<string> namesLayers  = {"Conv1", "Conv2", "Conv3", "FC1", "FC_Logits"}; 
    vector<size_t> sizesWeights = {5*5*1*32, 5*5*32*64, 5*5*64*128, 2048*512, 512*10}; 
    vector<size_t> sizesBiases = {32, 64, 128, 512, 10}; 
    vector<vector<size_t>> shapesWeights = {{5*5*1, 32}, {5*5*32, 64}, {5*5*64, 128}, {2048, 512}, {512, 10}}; 
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_weights; 
        int Z_weights; 
        vector<Scalar> Q_weights(sizesWeights[idx]); 
        
        ifstream fin_weights(WEIGHTFOLDER + namesLayers[idx] + "_weights.txt"); 
        if(!fin_weights)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_weights >> S_weights; 
        fin_weights >> Z_weights;
        for(size_t jdx = 0; jdx < sizesWeights[idx]; jdx++)
        {
            int tmp; 
            fin_weights >> tmp; 
            Q_weights[jdx] = tmp; 
        }
        fin_weights.close(); 

        NDArray weights(S_weights, Z_weights, shapesWeights[idx], Q_weights);  
        
        map_S_weights[namesLayers[idx]] = S_weights; 
        map_Z_weights[namesLayers[idx]] = Z_weights; 
        map_Q_weights[namesLayers[idx]] = Q_weights; 
        map_weights[namesLayers[idx]]   = weights; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_biases; 
        int Z_biases; 
        vector<int> Q_biases(sizesBiases[idx]); 
        
        ifstream fin_biases(WEIGHTFOLDER + namesLayers[idx] + "_biases.txt"); 
        if(!fin_biases)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_biases >> S_biases; 
        fin_biases >> Z_biases;
        for(size_t jdx = 0; jdx < sizesBiases[idx]; jdx++)
        {
            int tmp; 
            fin_biases >> tmp; 
            Q_biases[jdx] = tmp; 
        }
        fin_biases.close(); 
        
        map_S_biases[namesLayers[idx]] = S_biases; 
        map_Z_biases[namesLayers[idx]] = Z_biases; 
        map_Q_biases[namesLayers[idx]] = Q_biases; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_activations; 
        int Z_activations; 
        
        ifstream fin_activations(WEIGHTFOLDER + namesLayers[idx] + "_activations.txt"); 
        if(!fin_activations)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_activations >> S_activations; 
        fin_activations >> Z_activations;
        fin_activations.close(); 
        
        map_S_activations[namesLayers[idx]] = S_activations; 
        map_Z_activations[namesLayers[idx]] = Z_activations; 
    }
    vector<Ref<NDArray>> image_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        image_pool.push_back(new NDArray(S_input, Z_input, {28, 28, 1})); 
    }
    vector<Ref<Var>> input_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        input_pool.push_back(new Var(image_pool[idx])); 
    }
    
    cout << "Building Conv1: S = " << map_S_weights["Conv1"] << "; Z = " << map_Z_weights["Conv1"] << "; S_act = " << map_S_activations["Conv1"] << "; Z_act = " << map_Z_activations["Conv1"] << endl; 
    
    vector<Ref<Var>> weightsConv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv1_pool.push_back(new Var(map_weights["Conv1"])); 
    }
    vector<Ref<vector<int>>> biasesConv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv1_pool.push_back(new vector<int>(map_Q_biases["Conv1"])); 
    }
    vector<Ref<Var>> preconv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv1_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv1"], map_Z_activations["Conv1"], input_pool[idx], weightsConv1_pool[idx], biasesConv1_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv1_pool.push_back(MaxPool<NDArray>(preconv1_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv2: S = " << map_S_weights["Conv2"] << "; Z = " << map_Z_weights["Conv2"] << "; S_act = " << map_S_activations["Conv2"] << "; Z_act = " << map_Z_activations["Conv2"] << endl; 
    
    vector<Ref<Var>> weightsConv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv2_pool.push_back(new Var(map_weights["Conv2"])); 
    }
    vector<Ref<vector<int>>> biasesConv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv2_pool.push_back(new vector<int>(map_Q_biases["Conv2"])); 
    }
    vector<Ref<Var>> preconv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv2_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv2"], map_Z_activations["Conv2"], conv1_pool[idx], weightsConv2_pool[idx], biasesConv2_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv2_pool.push_back(MaxPool<NDArray>(preconv2_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv3: S = " << map_S_weights["Conv3"] << "; Z = " << map_Z_weights["Conv3"] << "; S_act = " << map_S_activations["Conv3"] << "; Z_act = " << map_Z_activations["Conv3"] << endl; 
    
    vector<Ref<Var>> weightsConv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv3_pool.push_back(new Var(map_weights["Conv3"])); 
    }
    vector<Ref<vector<int>>> biasesConv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv3_pool.push_back(new vector<int>(map_Q_biases["Conv3"])); 
    }
    vector<Ref<Var>> preconv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv3_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv3"], map_Z_activations["Conv3"], conv2_pool[idx], weightsConv3_pool[idx], biasesConv3_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv3_pool.push_back(MaxPool<NDArray>(preconv3_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    vector<Ref<Var>> flatten_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        flatten_pool.push_back(Flatten<NDArray>(conv3_pool[idx])); 
    }
    
    cout << "Building FC1: S = " << map_S_weights["FC1"] << "; Z = " << map_Z_weights["FC1"] << "; S_act = " << map_S_activations["FC1"] << "; Z_act = " << map_Z_activations["FC1"] << endl; 
    
    vector<Ref<Var>> weightsFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC1_pool.push_back(new Var(map_weights["FC1"])); 
    }
    vector<Ref<vector<int>>> biasesFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC1_pool.push_back(new vector<int>(map_Q_biases["FC1"])); 
    }
    vector<Ref<Var>> fc1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc1_pool.push_back(MADReLU<NDArray>(map_S_activations["FC1"], map_Z_activations["FC1"], flatten_pool[idx], weightsFC1_pool[idx], biasesFC1_pool[idx])); 
    }
    
    cout << "Building FC2: S = " << map_S_weights["FC_Logits"] << "; Z = " << map_Z_weights["FC_Logits"] << "; S_act = " << map_S_activations["FC_Logits"] << "; Z_act = " << map_Z_activations["FC_Logits"] << endl; 
    
    vector<Ref<Var>> weightsFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC2_pool.push_back(new Var(map_weights["FC_Logits"])); 
    }
    vector<Ref<vector<int>>> biasesFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC2_pool.push_back(new vector<int>(map_Q_biases["FC_Logits"])); 
    }
    vector<Ref<Var>> fc2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc2_pool.push_back(MAD<NDArray>(map_S_activations["FC_Logits"], map_Z_activations["FC_Logits"], fc1_pool[idx], weightsFC2_pool[idx], biasesFC2_pool[idx])); 
    }
    
    vector<vector<float>> images = getImagesFashion(); 
    vector<unsigned> labels = getLabelsFashion(); 
    
    float count = 0; 
    size_t TestSize = 10000;  
    for(size_t idx = 0; idx < TestSize; idx+=NUMTHREADS)
    {
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            vector<Scalar> Q_image(784); 
            for(size_t kdx = 0; kdx < 784; kdx++)
            {
                float tmp = images[idx+jdx][kdx]; 
                Q_image[kdx] = (Z_input + tmp / S_input > 0) ? round(Z_input + tmp / S_input) : 0;
            }
            image_pool[jdx]->set(Q_image); 
        }
        
        #pragma omp parallel for num_threads(NUMTHREADS)
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            fc2_pool[jdx]->evaluate(idx+1); 
        }
        
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            size_t label = fc2_pool[jdx]->value().posmax(); 
            if(label == labels[idx+jdx])
            {
                count++; 
            }
            //fc2_pool[jdx]->value().eval().print(); 
            cout << "Sample No." << (idx+jdx+1) << " " << "; Label: " << labels[idx+jdx] << ", Predicted: " << label << " -> " << ((label == labels[idx+jdx]) ? "Right" : "Wrong") << endl; 
        }
    }
    cout << "Accuray: " << (count / TestSize) << endl; 
    
    return 0; 
}

int testCIFAR10(const string &lutfile = "../Utils/LUT_HEAM.txt")
{   
    float S_input = 1.0 / 255.0; 
    int Z_input   = 0; 
    
    const size_t NUMTHREADS = 20; 
    const string WEIGHTFOLDER = "../Weights/CIFAR10/"; 
    
    NDArray::_loadLUT(lutfile); 
    
    unordered_map<string, float> map_S_weights; 
    unordered_map<string, int> map_Z_weights; 
    unordered_map<string, vector<Scalar>> map_Q_weights; 
    unordered_map<string, NDArray> map_weights; 
    
    unordered_map<string, float> map_S_biases; 
    unordered_map<string, int> map_Z_biases; 
    unordered_map<string, vector<int>> map_Q_biases; 
    
    unordered_map<string, float> map_S_activations; 
    unordered_map<string, int> map_Z_activations; 
    
    vector<string> namesLayers  = {"Conv1", "Conv2", "Conv3", "FC1", "FC_Logits"}; 
    vector<size_t> sizesWeights = {5*5*3*32, 5*5*32*64, 5*5*64*128, 2048*512, 512*10}; 
    vector<size_t> sizesBiases = {32, 64, 128, 512, 10}; 
    vector<vector<size_t>> shapesWeights = {{5*5*3, 32}, {5*5*32, 64}, {5*5*64, 128}, {2048, 512}, {512, 10}}; 
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_weights; 
        int Z_weights; 
        vector<Scalar> Q_weights(sizesWeights[idx]); 
        
        ifstream fin_weights(WEIGHTFOLDER + namesLayers[idx] + "_weights.txt"); 
        if(!fin_weights)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_weights >> S_weights; 
        fin_weights >> Z_weights;
        for(size_t jdx = 0; jdx < sizesWeights[idx]; jdx++)
        {
            int tmp; 
            fin_weights >> tmp; 
            Q_weights[jdx] = tmp; 
        }
        fin_weights.close(); 

        NDArray weights(S_weights, Z_weights, shapesWeights[idx], Q_weights);  
        
        map_S_weights[namesLayers[idx]] = S_weights; 
        map_Z_weights[namesLayers[idx]] = Z_weights; 
        map_Q_weights[namesLayers[idx]] = Q_weights; 
        map_weights[namesLayers[idx]]   = weights; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_biases; 
        int Z_biases; 
        vector<int> Q_biases(sizesBiases[idx]); 
        
        ifstream fin_biases(WEIGHTFOLDER + namesLayers[idx] + "_biases.txt"); 
        if(!fin_biases)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_biases >> S_biases; 
        fin_biases >> Z_biases;
        for(size_t jdx = 0; jdx < sizesBiases[idx]; jdx++)
        {
            int tmp; 
            fin_biases >> tmp; 
            Q_biases[jdx] = tmp; 
        }
        fin_biases.close(); 
        
        map_S_biases[namesLayers[idx]] = S_biases; 
        map_Z_biases[namesLayers[idx]] = Z_biases; 
        map_Q_biases[namesLayers[idx]] = Q_biases; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_activations; 
        int Z_activations; 
        
        ifstream fin_activations(WEIGHTFOLDER + namesLayers[idx] + "_activations.txt"); 
        if(!fin_activations)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_activations >> S_activations; 
        fin_activations >> Z_activations;
        fin_activations.close(); 
        
        map_S_activations[namesLayers[idx]] = S_activations; 
        map_Z_activations[namesLayers[idx]] = Z_activations; 
    }
    vector<Ref<NDArray>> image_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        image_pool.push_back(new NDArray(S_input, Z_input, {28, 28, 3})); 
    }
    vector<Ref<Var>> input_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        input_pool.push_back(new Var(image_pool[idx])); 
    }
    
    cout << "Building Conv1: S = " << map_S_weights["Conv1"] << "; Z = " << map_Z_weights["Conv1"] << "; S_act = " << map_S_activations["Conv1"] << "; Z_act = " << map_Z_activations["Conv1"] << endl; 
    
    vector<Ref<Var>> weightsConv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv1_pool.push_back(new Var(map_weights["Conv1"])); 
    }
    vector<Ref<vector<int>>> biasesConv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv1_pool.push_back(new vector<int>(map_Q_biases["Conv1"])); 
    }
    vector<Ref<Var>> preconv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv1_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv1"], map_Z_activations["Conv1"], input_pool[idx], weightsConv1_pool[idx], biasesConv1_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv1_pool.push_back(MaxPool<NDArray>(preconv1_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv2: S = " << map_S_weights["Conv2"] << "; Z = " << map_Z_weights["Conv2"] << "; S_act = " << map_S_activations["Conv2"] << "; Z_act = " << map_Z_activations["Conv2"] << endl; 
    
    vector<Ref<Var>> weightsConv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv2_pool.push_back(new Var(map_weights["Conv2"])); 
    }
    vector<Ref<vector<int>>> biasesConv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv2_pool.push_back(new vector<int>(map_Q_biases["Conv2"])); 
    }
    vector<Ref<Var>> preconv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv2_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv2"], map_Z_activations["Conv2"], conv1_pool[idx], weightsConv2_pool[idx], biasesConv2_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv2_pool.push_back(MaxPool<NDArray>(preconv2_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv3: S = " << map_S_weights["Conv3"] << "; Z = " << map_Z_weights["Conv3"] << "; S_act = " << map_S_activations["Conv3"] << "; Z_act = " << map_Z_activations["Conv3"] << endl; 
    
    vector<Ref<Var>> weightsConv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv3_pool.push_back(new Var(map_weights["Conv3"])); 
    }
    vector<Ref<vector<int>>> biasesConv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv3_pool.push_back(new vector<int>(map_Q_biases["Conv3"])); 
    }
    vector<Ref<Var>> preconv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv3_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv3"], map_Z_activations["Conv3"], conv2_pool[idx], weightsConv3_pool[idx], biasesConv3_pool[idx], {5, 5}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv3_pool.push_back(MaxPool<NDArray>(preconv3_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    vector<Ref<Var>> flatten_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        flatten_pool.push_back(Flatten<NDArray>(conv3_pool[idx])); 
    }
    
    cout << "Building FC1: S = " << map_S_weights["FC1"] << "; Z = " << map_Z_weights["FC1"] << "; S_act = " << map_S_activations["FC1"] << "; Z_act = " << map_Z_activations["FC1"] << endl; 
    
    vector<Ref<Var>> weightsFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC1_pool.push_back(new Var(map_weights["FC1"])); 
    }
    vector<Ref<vector<int>>> biasesFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC1_pool.push_back(new vector<int>(map_Q_biases["FC1"])); 
    }
    vector<Ref<Var>> fc1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc1_pool.push_back(MADReLU<NDArray>(map_S_activations["FC1"], map_Z_activations["FC1"], flatten_pool[idx], weightsFC1_pool[idx], biasesFC1_pool[idx])); 
    }
    
    Ref<Var> weightsConv1 = new Var(map_weights["Conv1"]); 
    Ref<vector<int>> biasesConv1 = new vector<int>(map_Q_biases["Conv1"]); 
    cout << "Weight Conv1: " << endl; 
    weightsConv1->value().print(); 
    weightsConv1->value().eval().print(); 
    NDArrayFloatCPU<float> _weightsConv1 = weightsConv1->value().eval(); 
    cout << "Bias Conv1: " << endl; 
    print(*biasesConv1); 
    NDArrayFloatCPU<float> _biasesConv1(sizesBiases[0]); 
    for(size_t idx  = 0; idx < _biasesConv1.size(); idx++)
    {
        _biasesConv1[idx] = map_S_biases["Conv1"] * (*biasesConv1)[idx]; 
    }
    Ref<Var> weightsConv2 = new Var(map_weights["Conv2"]); 
    Ref<vector<int>> biasesConv2 = new vector<int>(map_Q_biases["Conv2"]); 
    cout << "Weight Conv2: " << endl; 
    weightsConv2->value().print(); 
    weightsConv2->value().eval().print(); 
    NDArrayFloatCPU<float> _weightsConv2 = weightsConv2->value().eval(); 
    cout << "Bias Conv2: " << endl; 
    print(*biasesConv2); 
    NDArrayFloatCPU<float> _biasesConv2(sizesBiases[1]); 
    for(size_t idx  = 0; idx < _biasesConv2.size(); idx++)
    {
        _biasesConv2[idx] = map_S_biases["Conv2"] * (*biasesConv2)[idx]; 
    }
    Ref<Var> weightsConv3 = new Var(map_weights["Conv3"]); 
    Ref<vector<int>> biasesConv3 = new vector<int>(map_Q_biases["Conv3"]); 
    cout << "Weight Conv3: " << endl; 
    weightsConv3->value().print(); 
    weightsConv3->value().eval().print(); 
    NDArrayFloatCPU<float> _weightsConv3 = weightsConv3->value().eval(); 
    cout << "Bias Conv3: " << endl; 
    print(*biasesConv3); 
    NDArrayFloatCPU<float> _biasesConv3(sizesBiases[2]); 
    for(size_t idx  = 0; idx < _biasesConv3.size(); idx++)
    {
        _biasesConv3[idx] = map_S_biases["Conv3"] * (*biasesConv3)[idx]; 
    }
    Ref<Var> weightsFC1 = new Var(map_weights["FC1"]); 
    Ref<vector<int>> biasesFC1 = new vector<int>(map_Q_biases["FC1"]); 
    cout << "Weight FC1: " << endl; 
    weightsFC1->value().print(); 
    weightsFC1->value().eval().print(); 
    NDArrayFloatCPU<float> _weightsFC1 = weightsFC1->value().eval(); 
    cout << "Bias FC1: " << endl; 
    print(*biasesFC1); 
    NDArrayFloatCPU<float> _biasesFC1(sizesBiases[3]); 
    for(size_t idx  = 0; idx < _biasesFC1.size(); idx++)
    {
        _biasesFC1[idx] = map_S_biases["FC1"] * (*biasesFC1)[idx]; 
    }
    Ref<Var> weightsFC2 = new Var(map_weights["FC_Logits"]); 
    Ref<vector<int>> biasesFC2 = new vector<int>(map_Q_biases["FC_Logits"]); 
    cout << "Weight FC2: " << endl; 
    weightsFC2->value().print(); 
    weightsFC2->value().eval().print(); 
    NDArrayFloatCPU<float> _weightsFC2 = weightsFC2->value().eval(); 
    cout << "Bias FC2: " << endl; 
    print(*biasesFC2); 
    NDArrayFloatCPU<float> _biasesFC2(sizesBiases[4]); 
    for(size_t idx  = 0; idx < _biasesFC2.size(); idx++)
    {
        _biasesFC2[idx] = map_S_biases["FC_Logits"] * (*biasesFC2)[idx]; 
    }
    
    cout << "Building FC2: S = " << map_S_weights["FC_Logits"] << "; Z = " << map_Z_weights["FC_Logits"] << "; S_act = " << map_S_activations["FC_Logits"] << "; Z_act = " << map_Z_activations["FC_Logits"] << endl; 
    
    vector<Ref<Var>> weightsFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC2_pool.push_back(new Var(map_weights["FC_Logits"])); 
    }
    vector<Ref<vector<int>>> biasesFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC2_pool.push_back(new vector<int>(map_Q_biases["FC_Logits"])); 
    }
    vector<Ref<Var>> fc2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc2_pool.push_back(MAD<NDArray>(map_S_activations["FC_Logits"], map_Z_activations["FC_Logits"], fc1_pool[idx], weightsFC2_pool[idx], biasesFC2_pool[idx])); 
    }
    
    vector<vector<float>> images = getImagesCIFAR10(); 
    vector<unsigned> labels = getLabelsCIFAR10(); 
    
    float count = 0, _count = 0; 
    size_t TestSize = 10000;  
    for(size_t idx = 0; idx < TestSize; idx+=NUMTHREADS)
    {
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            vector<Scalar> Q_image(784*3); 
            for(size_t kdx = 0; kdx < 784*3; kdx++)
            {
                Q_image[kdx] = images[idx+jdx][kdx];
            }
            image_pool[jdx]->set(Q_image); 
        }
        
        int _labels[NUMTHREADS] = {0, }; 
        #pragma omp parallel for num_threads(NUMTHREADS)
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            fc2_pool[jdx]->evaluate(idx+1); 
            NDArrayFloatCPU<float> _image({28, 28, 3}, images[idx+jdx]); 
            _image = _image * (1.0 / 255.0); 
            NDArrayFloatCPU<float> _conv1 = (_image.im2col({5, 5}, {1, 1}, true) * _weightsConv1).reshape({28, 28, 32}).addChWise(_biasesConv1).ReLU().maxPool({2, 2}, {2, 2}, true); 
    //         std::cout << _conv1.shape()[0] << ", " << _conv1.shape()[1] << ", " << _conv1.shape()[2]  << std::endl; 
            NDArrayFloatCPU<float> _conv2 = (_conv1.im2col({5, 5}, {1, 1}, true) * _weightsConv2).reshape({14, 14, 64}).addChWise(_biasesConv2).ReLU().maxPool({2, 2}, {2, 2}, true); 
    //         std::cout << _conv2.shape()[0] << ", " << _conv2.shape()[1] << ", " << _conv2.shape()[2]  << std::endl; 
            NDArrayFloatCPU<float> _conv3 = (_conv2.im2col({5, 5}, {1, 1}, true) * _weightsConv3).reshape({7, 7, 128}).addChWise(_biasesConv3).ReLU().maxPool({2, 2}, {2, 2}, true); 
    //         std::cout << _conv3.shape()[0] << ", " << _conv3.shape()[1] << ", " << _conv3.shape()[2]  << std::endl; 
            NDArrayFloatCPU<float> _fc1 = (_conv3.reshape({1, 2048}) * _weightsFC1 +_biasesFC1.reshape({1, 512})).ReLU(); 
            NDArrayFloatCPU<float> _fc2 = _fc1 * _weightsFC2 +_biasesFC2.reshape({1, 10}); 
            _labels[jdx] = _fc2.posmax(); 
        }
        
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            size_t label = fc2_pool[jdx]->value().posmax(); 
            size_t _label = _labels[jdx]; 
            if(label == labels[idx+jdx])
            {
                count++; 
            }
            if(_label == labels[idx+jdx])
            {
                _count++; 
            }
            //fc2_pool[jdx]->value().eval().print(); 
            cout << "Sample No." << (idx+jdx+1) << " " << "; Label: " << labels[idx+jdx] << ", Predicted: " << label << " / " << _label  << " -> " << ((label == labels[idx+jdx]) ? "Right" : "Wrong") << " / "  << ((_label == labels[idx+jdx]) ? "Right" : "Wrong") << "; Accuracy: " << (count / (idx+jdx+1)) <<   " / " << (_count / (idx+jdx+1)) << endl; 
        }
    }
    cout << "Accuray: " << (count / TestSize) << endl; 
    cout << "_Accuray: " << (_count / TestSize) << endl; 
    
    return 0; 
}

int testCIFAR10AlexNet(const string &lutfile = "../Utils/LUT_HEAM.txt") 
{   
    float S_input = 1.0 / 255.0; 
    int Z_input   = 0; 
    
    const size_t NUMTHREADS = 20; 
    const string WEIGHTFOLDER = "../Weights/CIFAR10/"; 
    
    NDArray::_loadLUT(lutfile); 
    
    unordered_map<string, float> map_S_weights; 
    unordered_map<string, int> map_Z_weights; 
    unordered_map<string, vector<Scalar>> map_Q_weights; 
    unordered_map<string, NDArray> map_weights; 
    
    unordered_map<string, float> map_S_biases; 
    unordered_map<string, int> map_Z_biases; 
    unordered_map<string, vector<int>> map_Q_biases; 
    
    unordered_map<string, float> map_S_activations; 
    unordered_map<string, int> map_Z_activations; 
    
    vector<string> namesLayers  = {"Conv1a", "Conv1b", "Conv2a", "Conv2b", "Conv3a", "Conv3b", "Conv4a", "Conv4b", "FC1", "FC2", "FC_Logits"}; 
    vector<size_t> sizesWeights = {3*3*3*96, 3*3*96*96,  3*3*96*256, 3*3*256*256, 3*3*256*384, 3*3*384*384, 3*3*384*256, 3*3*256*256, 4096*4096, 4096*4096, 4096*10}; 
    vector<size_t> sizesBiases = {96, 96, 256, 256, 384, 384, 256, 256, 4096, 4096, 10}; 
    vector<vector<size_t>> shapesWeights = {{3*3*3, 96}, {3*3*96, 96}, {3*3*96, 256}, {3*3*256, 256}, {3*3*256, 384}, {3*3*384, 384}, {3*3*384, 256}, {3*3*256, 256}, {4096, 4096}, {4096, 4096}, {4096, 10}}; 
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_weights; 
        int Z_weights; 
        vector<Scalar> Q_weights(sizesWeights[idx]); 
        
        ifstream fin_weights(WEIGHTFOLDER + namesLayers[idx] + "_weights.txt"); 
        if(!fin_weights)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_weights >> S_weights; 
        fin_weights >> Z_weights;
        for(size_t jdx = 0; jdx < sizesWeights[idx]; jdx++)
        {
            int tmp; 
            fin_weights >> tmp; 
            Q_weights[jdx] = tmp; 
        }
        fin_weights.close(); 

        NDArray weights(S_weights, Z_weights, shapesWeights[idx], Q_weights);  
        
        map_S_weights[namesLayers[idx]] = S_weights; 
        map_Z_weights[namesLayers[idx]] = Z_weights; 
        map_Q_weights[namesLayers[idx]] = Q_weights; 
        map_weights[namesLayers[idx]]   = weights; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_biases; 
        int Z_biases; 
        vector<int> Q_biases(sizesBiases[idx]); 
        
        ifstream fin_biases(WEIGHTFOLDER + namesLayers[idx] + "_biases.txt"); 
        if(!fin_biases)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_biases >> S_biases; 
        fin_biases >> Z_biases;
        for(size_t jdx = 0; jdx < sizesBiases[idx]; jdx++)
        {
            int tmp; 
            fin_biases >> tmp; 
            Q_biases[jdx] = tmp; 
        }
        fin_biases.close(); 
        
        map_S_biases[namesLayers[idx]] = S_biases; 
        map_Z_biases[namesLayers[idx]] = Z_biases; 
        map_Q_biases[namesLayers[idx]] = Q_biases; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_activations; 
        int Z_activations; 
        
        ifstream fin_activations(WEIGHTFOLDER + namesLayers[idx] + "_activations.txt"); 
        if(!fin_activations)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_activations >> S_activations; 
        fin_activations >> Z_activations;
        fin_activations.close(); 
        
        map_S_activations[namesLayers[idx]] = S_activations; 
        map_Z_activations[namesLayers[idx]] = Z_activations; 
    }
    vector<Ref<NDArray>> image_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        image_pool.push_back(new NDArray(S_input, Z_input, {28, 28, 3})); 
    }
    vector<Ref<Var>> input_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        input_pool.push_back(new Var(image_pool[idx])); 
    }
    
    assert((namesLayers.size() == sizesWeights.size()) && (namesLayers.size() == sizesBiases.size()) && (namesLayers.size() == shapesWeights.size())); 
    
    cout << "Building Conv1a: S = " << map_S_weights["Conv1a"] << "; Z = " << map_Z_weights["Conv1a"] << "; S_act = " << map_S_activations["Conv1a"] << "; Z_act = " << map_Z_activations["Conv1a"] << endl; 
    
    vector<Ref<Var>> weightsConv1a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv1a_pool.push_back(new Var(map_weights["Conv1a"])); 
    }
    vector<Ref<vector<int>>> biasesConv1a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv1a_pool.push_back(new vector<int>(map_Q_biases["Conv1a"])); 
    }
    vector<Ref<Var>> conv1a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv1a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv1a"], map_Z_activations["Conv1a"], input_pool[idx], weightsConv1a_pool[idx], biasesConv1a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv1b: S = " << map_S_weights["Conv1b"] << "; Z = " << map_Z_weights["Conv1b"] << "; S_act = " << map_S_activations["Conv1b"] << "; Z_act = " << map_Z_activations["Conv1b"] << endl; 
    
    vector<Ref<Var>> weightsConv1b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv1b_pool.push_back(new Var(map_weights["Conv1b"])); 
    }
    vector<Ref<vector<int>>> biasesConv1b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv1b_pool.push_back(new vector<int>(map_Q_biases["Conv1b"])); 
    }
    vector<Ref<Var>> conv1b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv1b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv1b"], map_Z_activations["Conv1b"], conv1a_pool[idx], weightsConv1b_pool[idx], biasesConv1b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv2a: S = " << map_S_weights["Conv2a"] << "; Z = " << map_Z_weights["Conv2a"] << "; S_act = " << map_S_activations["Conv2a"] << "; Z_act = " << map_Z_activations["Conv2a"] << endl; 
    
    vector<Ref<Var>> weightsConv2a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv2a_pool.push_back(new Var(map_weights["Conv2a"])); 
    }
    vector<Ref<vector<int>>> biasesConv2a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv2a_pool.push_back(new vector<int>(map_Q_biases["Conv2a"])); 
    }
    vector<Ref<Var>> conv2a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv2a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv2a"], map_Z_activations["Conv2a"], conv1b_pool[idx], weightsConv2a_pool[idx], biasesConv2a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv2b: S = " << map_S_weights["Conv2b"] << "; Z = " << map_Z_weights["Conv2b"] << "; S_act = " << map_S_activations["Conv2b"] << "; Z_act = " << map_Z_activations["Conv2b"] << endl; 
    
    vector<Ref<Var>> weightsConv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv2b_pool.push_back(new Var(map_weights["Conv2b"])); 
    }
    vector<Ref<vector<int>>> biasesConv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv2b_pool.push_back(new vector<int>(map_Q_biases["Conv2b"])); 
    }
    vector<Ref<Var>> preconv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv2b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv2b"], map_Z_activations["Conv2b"], conv2a_pool[idx], weightsConv2b_pool[idx], biasesConv2b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv2b_pool.push_back(MaxPool<NDArray>(preconv2b_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv3a: S = " << map_S_weights["Conv3a"] << "; Z = " << map_Z_weights["Conv3a"] << "; S_act = " << map_S_activations["Conv3a"] << "; Z_act = " << map_Z_activations["Conv3a"] << endl; 
    
    vector<Ref<Var>> weightsConv3a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv3a_pool.push_back(new Var(map_weights["Conv3a"])); 
    }
    vector<Ref<vector<int>>> biasesConv3a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv3a_pool.push_back(new vector<int>(map_Q_biases["Conv3a"])); 
    }
    vector<Ref<Var>> conv3a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv3a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv3a"], map_Z_activations["Conv3a"], conv2b_pool[idx], weightsConv3a_pool[idx], biasesConv3a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv3b: S = " << map_S_weights["Conv3b"] << "; Z = " << map_Z_weights["Conv3b"] << "; S_act = " << map_S_activations["Conv3b"] << "; Z_act = " << map_Z_activations["Conv3b"] << endl; 
    
    vector<Ref<Var>> weightsConv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv3b_pool.push_back(new Var(map_weights["Conv3b"])); 
    }
    vector<Ref<vector<int>>> biasesConv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv3b_pool.push_back(new vector<int>(map_Q_biases["Conv3b"])); 
    }
    vector<Ref<Var>> preconv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv3b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv3b"], map_Z_activations["Conv3b"], conv3a_pool[idx], weightsConv3b_pool[idx], biasesConv3b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv3b_pool.push_back(MaxPool<NDArray>(preconv3b_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv4a: S = " << map_S_weights["Conv4a"] << "; Z = " << map_Z_weights["Conv4a"] << "; S_act = " << map_S_activations["Conv4a"] << "; Z_act = " << map_Z_activations["Conv4a"] << endl; 
    
    vector<Ref<Var>> weightsConv4a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv4a_pool.push_back(new Var(map_weights["Conv4a"])); 
    }
    vector<Ref<vector<int>>> biasesConv4a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv4a_pool.push_back(new vector<int>(map_Q_biases["Conv4a"])); 
    }
    vector<Ref<Var>> conv4a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv4a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv4a"], map_Z_activations["Conv4a"], conv3b_pool[idx], weightsConv4a_pool[idx], biasesConv4a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv4b: S = " << map_S_weights["Conv4b"] << "; Z = " << map_Z_weights["Conv4b"] << "; S_act = " << map_S_activations["Conv4b"] << "; Z_act = " << map_Z_activations["Conv4b"] << endl; 
    
    vector<Ref<Var>> weightsConv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv4b_pool.push_back(new Var(map_weights["Conv4b"])); 
    }
    vector<Ref<vector<int>>> biasesConv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv4b_pool.push_back(new vector<int>(map_Q_biases["Conv4b"])); 
    }
    vector<Ref<Var>> preconv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv4b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv4b"], map_Z_activations["Conv4b"], conv4a_pool[idx], weightsConv4b_pool[idx], biasesConv4b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv4b_pool.push_back(MaxPool<NDArray>(preconv4b_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    vector<Ref<Var>> flatten_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        flatten_pool.push_back(Flatten<NDArray>(conv4b_pool[idx])); 
    }
    
    cout << "Building FC1: S = " << map_S_weights["FC1"] << "; Z = " << map_Z_weights["FC1"] << "; S_act = " << map_S_activations["FC1"] << "; Z_act = " << map_Z_activations["FC1"] << endl; 
    
    vector<Ref<Var>> weightsFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC1_pool.push_back(new Var(map_weights["FC1"])); 
    }
    vector<Ref<vector<int>>> biasesFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC1_pool.push_back(new vector<int>(map_Q_biases["FC1"])); 
    }
    vector<Ref<Var>> fc1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc1_pool.push_back(MADReLU<NDArray>(map_S_activations["FC1"], map_Z_activations["FC1"], flatten_pool[idx], weightsFC1_pool[idx], biasesFC1_pool[idx])); 
    }
    
    cout << "Building FC2: S = " << map_S_weights["FC2"] << "; Z = " << map_Z_weights["FC2"] << "; S_act = " << map_S_activations["FC2"] << "; Z_act = " << map_Z_activations["FC2"] << endl; 
    
    vector<Ref<Var>> weightsFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC2_pool.push_back(new Var(map_weights["FC2"])); 
    }
    vector<Ref<vector<int>>> biasesFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC2_pool.push_back(new vector<int>(map_Q_biases["FC2"])); 
    }
    vector<Ref<Var>> fc2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc2_pool.push_back(MADReLU<NDArray>(map_S_activations["FC2"], map_Z_activations["FC2"], fc1_pool[idx], weightsFC2_pool[idx], biasesFC2_pool[idx])); 
    }
    
    cout << "Building FC_Logits: S = " << map_S_weights["FC_Logits"] << "; Z = " << map_Z_weights["FC_Logits"] << "; S_act = " << map_S_activations["FC_Logits"] << "; Z_act = " << map_Z_activations["FC_Logits"] << endl; 
    
    vector<Ref<Var>> weightsFC3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC3_pool.push_back(new Var(map_weights["FC_Logits"])); 
    }
    vector<Ref<vector<int>>> biasesFC3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC3_pool.push_back(new vector<int>(map_Q_biases["FC_Logits"])); 
    }
    vector<Ref<Var>> fc3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc3_pool.push_back(MAD<NDArray>(map_S_activations["FC_Logits"], map_Z_activations["FC_Logits"], fc2_pool[idx], weightsFC3_pool[idx], biasesFC3_pool[idx])); 
    }
    
    vector<vector<float>> images = getImagesCIFAR10(); 
    vector<unsigned> labels = getLabelsCIFAR10(); 
    
    float count = 0; 
    size_t TestSize = 10000;  
    for(size_t idx = 0; idx < TestSize; idx+=NUMTHREADS)
    {
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            vector<Scalar> Q_image(784*3); 
            for(size_t kdx = 0; kdx < 784*3; kdx++)
            {
                Q_image[kdx] = images[idx+jdx][kdx];
            }
            image_pool[jdx]->set(Q_image); 
        }
        
        #pragma omp parallel for num_threads(NUMTHREADS)
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            fc3_pool[jdx]->evaluate(idx+1); 
        }
        
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            size_t label = fc3_pool[jdx]->value().posmax(); 
            if(label == labels[idx+jdx])
            {
                count++; 
            }
//             fc2_pool[jdx]->value().eval().print(); 
            cout << "Sample No." << (idx+jdx+1) << " " << "; Label: " << labels[idx+jdx] << ", Predicted: " << label << " / " << labels[idx+jdx] << " -> " << ((label == labels[idx+jdx]) ? "Right" : "Wrong") << "; Accuracy: " << (count / (idx+jdx+1)) << endl; 
        }
    }
    cout << "Accuray: " << (count / TestSize) << endl; 
    
    return 0; 
}

int testCIFAR10AlexNetExport(const string &lutfile = "../Utils/LUT_HEAM.txt")
{   
    float S_input = 1.0 / 255.0; 
    int Z_input   = 0; 
    
    const size_t NUMTHREADS = 20; 
    const string WEIGHTFOLDER = "../Weights/CIFAR10/"; 
    
    NDArray::_loadLUT(lutfile); 
    
    unordered_map<string, float> map_S_weights; 
    unordered_map<string, int> map_Z_weights; 
    unordered_map<string, vector<Scalar>> map_Q_weights; 
    unordered_map<string, NDArray> map_weights; 
    
    unordered_map<string, float> map_S_biases; 
    unordered_map<string, int> map_Z_biases; 
    unordered_map<string, vector<int>> map_Q_biases; 
    
    unordered_map<string, float> map_S_activations; 
    unordered_map<string, int> map_Z_activations; 
    
    vector<string> namesLayers  = {"Conv1a", "Conv1b", "Conv2a", "Conv2b", "Conv3a", "Conv3b", "Conv4a", "Conv4b", "FC1", "FC2", "FC_Logits"}; 
    vector<size_t> sizesWeights = {3*3*3*96, 3*3*96*96,  3*3*96*256, 3*3*256*256, 3*3*256*384, 3*3*384*384, 3*3*384*256, 3*3*256*256, 4096*4096, 4096*4096, 4096*10}; 
    vector<size_t> sizesBiases = {96, 96, 256, 256, 384, 384, 256, 256, 4096, 4096, 10}; 
    vector<vector<size_t>> shapesWeights = {{3*3*3, 96}, {3*3*96, 96}, {3*3*96, 256}, {3*3*256, 256}, {3*3*256, 384}, {3*3*384, 384}, {3*3*384, 256}, {3*3*256, 256}, {4096, 4096}, {4096, 4096}, {4096, 10}}; 
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_weights; 
        int Z_weights; 
        vector<Scalar> Q_weights(sizesWeights[idx]); 
        
        ifstream fin_weights(WEIGHTFOLDER + namesLayers[idx] + "_weights.txt"); 
        if(!fin_weights)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_weights >> S_weights; 
        fin_weights >> Z_weights;
        for(size_t jdx = 0; jdx < sizesWeights[idx]; jdx++)
        {
            int tmp; 
            fin_weights >> tmp; 
            Q_weights[jdx] = tmp; 
        }
        fin_weights.close(); 

        NDArray weights(S_weights, Z_weights, shapesWeights[idx], Q_weights);  
        
        map_S_weights[namesLayers[idx]] = S_weights; 
        map_Z_weights[namesLayers[idx]] = Z_weights; 
        map_Q_weights[namesLayers[idx]] = Q_weights; 
        map_weights[namesLayers[idx]]   = weights; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_biases; 
        int Z_biases; 
        vector<int> Q_biases(sizesBiases[idx]); 
        
        ifstream fin_biases(WEIGHTFOLDER + namesLayers[idx] + "_biases.txt"); 
        if(!fin_biases)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_biases >> S_biases; 
        fin_biases >> Z_biases;
        for(size_t jdx = 0; jdx < sizesBiases[idx]; jdx++)
        {
            int tmp; 
            fin_biases >> tmp; 
            Q_biases[jdx] = tmp; 
        }
        fin_biases.close(); 
        
        map_S_biases[namesLayers[idx]] = S_biases; 
        map_Z_biases[namesLayers[idx]] = Z_biases; 
        map_Q_biases[namesLayers[idx]] = Q_biases; 
    }
    
    for(size_t idx = 0; idx < namesLayers.size(); idx++)
    {
        float S_activations; 
        int Z_activations; 
        
        ifstream fin_activations(WEIGHTFOLDER + namesLayers[idx] + "_activations.txt"); 
        if(!fin_activations)
        {
            cerr << "ERROR: failed to open the file. " << endl; 
            exit(1); 
        }
        fin_activations >> S_activations; 
        fin_activations >> Z_activations;
        fin_activations.close(); 
        
        map_S_activations[namesLayers[idx]] = S_activations; 
        map_Z_activations[namesLayers[idx]] = Z_activations; 
    }
    vector<Ref<NDArray>> image_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        image_pool.push_back(new NDArray(S_input, Z_input, {28, 28, 3})); 
    }
    vector<Ref<Var>> input_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        input_pool.push_back(new Var(image_pool[idx])); 
    }
    
    assert((namesLayers.size() == sizesWeights.size()) && (namesLayers.size() == sizesBiases.size()) && (namesLayers.size() == shapesWeights.size())); 
    
    cout << "Building Conv1a: S = " << map_S_weights["Conv1a"] << "; Z = " << map_Z_weights["Conv1a"] << "; S_act = " << map_S_activations["Conv1a"] << "; Z_act = " << map_Z_activations["Conv1a"] << endl; 
    
    vector<Ref<Var>> weightsConv1a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv1a_pool.push_back(new Var(map_weights["Conv1a"])); 
    }
    vector<Ref<vector<int>>> biasesConv1a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv1a_pool.push_back(new vector<int>(map_Q_biases["Conv1a"])); 
    }
    vector<Ref<Var>> conv1a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv1a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv1a"], map_Z_activations["Conv1a"], input_pool[idx], weightsConv1a_pool[idx], biasesConv1a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv1b: S = " << map_S_weights["Conv1b"] << "; Z = " << map_Z_weights["Conv1b"] << "; S_act = " << map_S_activations["Conv1b"] << "; Z_act = " << map_Z_activations["Conv1b"] << endl; 
    
    vector<Ref<Var>> weightsConv1b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv1b_pool.push_back(new Var(map_weights["Conv1b"])); 
    }
    vector<Ref<vector<int>>> biasesConv1b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv1b_pool.push_back(new vector<int>(map_Q_biases["Conv1b"])); 
    }
    vector<Ref<Var>> conv1b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv1b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv1b"], map_Z_activations["Conv1b"], conv1a_pool[idx], weightsConv1b_pool[idx], biasesConv1b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv2a: S = " << map_S_weights["Conv2a"] << "; Z = " << map_Z_weights["Conv2a"] << "; S_act = " << map_S_activations["Conv2a"] << "; Z_act = " << map_Z_activations["Conv2a"] << endl; 
    
    vector<Ref<Var>> weightsConv2a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv2a_pool.push_back(new Var(map_weights["Conv2a"])); 
    }
    vector<Ref<vector<int>>> biasesConv2a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv2a_pool.push_back(new vector<int>(map_Q_biases["Conv2a"])); 
    }
    vector<Ref<Var>> conv2a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv2a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv2a"], map_Z_activations["Conv2a"], conv1b_pool[idx], weightsConv2a_pool[idx], biasesConv2a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv2b: S = " << map_S_weights["Conv2b"] << "; Z = " << map_Z_weights["Conv2b"] << "; S_act = " << map_S_activations["Conv2b"] << "; Z_act = " << map_Z_activations["Conv2b"] << endl; 
    
    vector<Ref<Var>> weightsConv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv2b_pool.push_back(new Var(map_weights["Conv2b"])); 
    }
    vector<Ref<vector<int>>> biasesConv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv2b_pool.push_back(new vector<int>(map_Q_biases["Conv2b"])); 
    }
    vector<Ref<Var>> preconv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv2b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv2b"], map_Z_activations["Conv2b"], conv2a_pool[idx], weightsConv2b_pool[idx], biasesConv2b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv2b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv2b_pool.push_back(MaxPool<NDArray>(preconv2b_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv3a: S = " << map_S_weights["Conv3a"] << "; Z = " << map_Z_weights["Conv3a"] << "; S_act = " << map_S_activations["Conv3a"] << "; Z_act = " << map_Z_activations["Conv3a"] << endl; 
    
    vector<Ref<Var>> weightsConv3a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv3a_pool.push_back(new Var(map_weights["Conv3a"])); 
    }
    vector<Ref<vector<int>>> biasesConv3a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv3a_pool.push_back(new vector<int>(map_Q_biases["Conv3a"])); 
    }
    vector<Ref<Var>> conv3a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv3a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv3a"], map_Z_activations["Conv3a"], conv2b_pool[idx], weightsConv3a_pool[idx], biasesConv3a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv3b: S = " << map_S_weights["Conv3b"] << "; Z = " << map_Z_weights["Conv3b"] << "; S_act = " << map_S_activations["Conv3b"] << "; Z_act = " << map_Z_activations["Conv3b"] << endl; 
    
    vector<Ref<Var>> weightsConv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv3b_pool.push_back(new Var(map_weights["Conv3b"])); 
    }
    vector<Ref<vector<int>>> biasesConv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv3b_pool.push_back(new vector<int>(map_Q_biases["Conv3b"])); 
    }
    vector<Ref<Var>> preconv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv3b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv3b"], map_Z_activations["Conv3b"], conv3a_pool[idx], weightsConv3b_pool[idx], biasesConv3b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv3b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv3b_pool.push_back(MaxPool<NDArray>(preconv3b_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    cout << "Building Conv4a: S = " << map_S_weights["Conv4a"] << "; Z = " << map_Z_weights["Conv4a"] << "; S_act = " << map_S_activations["Conv4a"] << "; Z_act = " << map_Z_activations["Conv4a"] << endl; 
    
    vector<Ref<Var>> weightsConv4a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv4a_pool.push_back(new Var(map_weights["Conv4a"])); 
    }
    vector<Ref<vector<int>>> biasesConv4a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv4a_pool.push_back(new vector<int>(map_Q_biases["Conv4a"])); 
    }
    vector<Ref<Var>> conv4a_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv4a_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv4a"], map_Z_activations["Conv4a"], conv3b_pool[idx], weightsConv4a_pool[idx], biasesConv4a_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    
    cout << "Building Conv4b: S = " << map_S_weights["Conv4b"] << "; Z = " << map_Z_weights["Conv4b"] << "; S_act = " << map_S_activations["Conv4b"] << "; Z_act = " << map_Z_activations["Conv4b"] << endl; 
    
    vector<Ref<Var>> weightsConv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsConv4b_pool.push_back(new Var(map_weights["Conv4b"])); 
    }
    vector<Ref<vector<int>>> biasesConv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesConv4b_pool.push_back(new vector<int>(map_Q_biases["Conv4b"])); 
    }
    vector<Ref<Var>> preconv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        preconv4b_pool.push_back(Conv2DReLU<NDArray>(map_S_activations["Conv4b"], map_Z_activations["Conv4b"], conv4a_pool[idx], weightsConv4b_pool[idx], biasesConv4b_pool[idx], {3, 3}, {1, 1}, true)); 
    }
    vector<Ref<Var>> conv4b_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        conv4b_pool.push_back(MaxPool<NDArray>(preconv4b_pool[idx], {2, 2}, {2, 2}, true)); 
    }
    
    vector<Ref<Var>> flatten_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        flatten_pool.push_back(Flatten<NDArray>(conv4b_pool[idx])); 
    }
    
    cout << "Building FC1: S = " << map_S_weights["FC1"] << "; Z = " << map_Z_weights["FC1"] << "; S_act = " << map_S_activations["FC1"] << "; Z_act = " << map_Z_activations["FC1"] << endl; 
    
    vector<Ref<Var>> weightsFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC1_pool.push_back(new Var(map_weights["FC1"])); 
    }
    vector<Ref<vector<int>>> biasesFC1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC1_pool.push_back(new vector<int>(map_Q_biases["FC1"])); 
    }
    vector<Ref<Var>> fc1_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc1_pool.push_back(MADReLU<NDArray>(map_S_activations["FC1"], map_Z_activations["FC1"], flatten_pool[idx], weightsFC1_pool[idx], biasesFC1_pool[idx])); 
    }
    
    cout << "Building FC2: S = " << map_S_weights["FC2"] << "; Z = " << map_Z_weights["FC2"] << "; S_act = " << map_S_activations["FC2"] << "; Z_act = " << map_Z_activations["FC2"] << endl; 
    
    vector<Ref<Var>> weightsFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC2_pool.push_back(new Var(map_weights["FC2"])); 
    }
    vector<Ref<vector<int>>> biasesFC2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC2_pool.push_back(new vector<int>(map_Q_biases["FC2"])); 
    }
    vector<Ref<Var>> fc2_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc2_pool.push_back(MADReLU<NDArray>(map_S_activations["FC2"], map_Z_activations["FC2"], fc1_pool[idx], weightsFC2_pool[idx], biasesFC2_pool[idx])); 
    }
    
    cout << "Building FC_Logits: S = " << map_S_weights["FC_Logits"] << "; Z = " << map_Z_weights["FC_Logits"] << "; S_act = " << map_S_activations["FC_Logits"] << "; Z_act = " << map_Z_activations["FC_Logits"] << endl; 
    
    vector<Ref<Var>> weightsFC3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        weightsFC3_pool.push_back(new Var(map_weights["FC_Logits"])); 
    }
    vector<Ref<vector<int>>> biasesFC3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        biasesFC3_pool.push_back(new vector<int>(map_Q_biases["FC_Logits"])); 
    }
    vector<Ref<Var>> fc3_pool; 
    for(size_t idx = 0; idx < NUMTHREADS; idx++)
    {
        fc3_pool.push_back(MAD<NDArray>(map_S_activations["FC_Logits"], map_Z_activations["FC_Logits"], fc2_pool[idx], weightsFC3_pool[idx], biasesFC3_pool[idx])); 
    }
    
    vector<vector<float>> images = getImagesCIFAR10(); 
    vector<unsigned> labels = getLabelsCIFAR10(); 
    
    float count = 0; 
    size_t TestSize = 1000;  
    
    vector<vector<unsigned>> layer0(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer1(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer2(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer3(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer4(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer5(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer6(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer7(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer8(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer9(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer10(TestSize, vector<unsigned>()); 
    vector<vector<unsigned>> layer11(TestSize, vector<unsigned>()); 
    
    for(size_t idx = 0; idx < TestSize; idx+=NUMTHREADS)
    {
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            vector<Scalar> Q_image(784*3); 
            for(size_t kdx = 0; kdx < 784*3; kdx++)
            {
                Q_image[kdx] = images[idx+jdx][kdx];
            }
            image_pool[jdx]->set(Q_image); 
        }
        
        #pragma omp parallel for num_threads(NUMTHREADS)
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            fc3_pool[jdx]->evaluate(idx+1); 
        }
        
        for(size_t jdx = 0; jdx < NUMTHREADS; jdx++)
        {
            size_t index = idx + jdx; 
            auto temp0 = image_pool[jdx].value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp0.size(); kdx++)
            {
                layer0[index].push_back(temp0[kdx]); 
            }
            auto temp1 = conv1a_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp1.size(); kdx++)
            {
                layer1[index].push_back(temp1[kdx]); 
            }
            auto temp2 = conv1b_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp2.size(); kdx++)
            {
                layer2[index].push_back(temp2[kdx]); 
            }
            auto temp3 = conv2a_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp3.size(); kdx++)
            {
                layer3[index].push_back(temp3[kdx]); 
            }
            auto temp4 = conv2b_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp4.size(); kdx++)
            {
                layer4[index].push_back(temp4[kdx]); 
            }
            auto temp5 = conv3a_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp5.size(); kdx++)
            {
                layer5[index].push_back(temp5[kdx]); 
            }
            auto temp6 = conv3b_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp6.size(); kdx++)
            {
                layer6[index].push_back(temp6[kdx]); 
            }
            auto temp7 = conv4a_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp7.size(); kdx++)
            {
                layer7[index].push_back(temp7[kdx]); 
            }
            auto temp8 = conv4b_pool[jdx].value().value().im2col({3, 3}, {1, 1}, true); 
            for(size_t kdx = 0; kdx < temp8.size(); kdx++)
            {
                layer8[index].push_back(temp8[kdx]); 
            }
            auto temp9 = fc1_pool[jdx].value().value(); 
            for(size_t kdx = 0; kdx < temp9.size(); kdx++)
            {
                layer9[index].push_back(temp9[kdx]); 
            }
            auto temp10 = fc2_pool[jdx].value().value(); 
            for(size_t kdx = 0; kdx < temp10.size(); kdx++)
            {
                layer10[index].push_back(temp10[kdx]); 
            }
            auto temp11 = fc3_pool[jdx].value().value(); 
            for(size_t kdx = 0; kdx < temp11.size(); kdx++)
            {
                layer11[index].push_back(temp11[kdx]); 
            }
            
            size_t label = fc3_pool[jdx]->value().posmax(); 
            if(label == labels[idx+jdx])
            {
                count++; 
            }
//             fc2_pool[jdx]->value().eval().print(); 
            cout << "Sample No." << (idx+jdx+1) << " " << "; Label: " << labels[idx+jdx] << ", Predicted: " << label << " / " << labels[idx+jdx] << " -> " << ((label == labels[idx+jdx]) ? "Right" : "Wrong") << "; Accuracy: " << (count / (idx+jdx+1)) << endl; 
        }
    }
    cout << "Accuray: " << (count / TestSize) << endl; 
    
    ofstream fout0("images_AlexNet.txt"); 
    ofstream fout1("layer1_AlexNet.txt"); 
    ofstream fout2("layer2_AlexNet.txt"); 
    ofstream fout3("layer3_AlexNet.txt"); 
    ofstream fout4("layer4_AlexNet.txt"); 
    ofstream fout5("layer5_AlexNet.txt"); 
    ofstream fout6("layer6_AlexNet.txt"); 
    ofstream fout7("layer7_AlexNet.txt"); 
    ofstream fout8("layer8_AlexNet.txt"); 
    ofstream fout9("layer9_AlexNet.txt"); 
    ofstream fout10("layer10_AlexNet.txt"); 
    ofstream fout11("layer11_AlexNet.txt"); 
    
    for(const auto &line: layer0)
    {
        for(const auto &elem: line)
        {
            fout0 << elem << " "; 
        }
        fout0 << endl; 
    }
    for(const auto &line: layer1)
    {
        for(const auto &elem: line)
        {
            fout1 << elem << " "; 
        }
        fout1 << endl; 
    }
    for(const auto &line: layer2)
    {
        for(const auto &elem: line)
        {
            fout2 << elem << " "; 
        }
        fout2 << endl; 
    }
    for(const auto &line: layer3)
    {
        for(const auto &elem: line)
        {
            fout3 << elem << " "; 
        }
        fout3 << endl; 
    }
    for(const auto &line: layer4)
    {
        for(const auto &elem: line)
        {
            fout4 << elem << " "; 
        }
        fout4 << endl; 
    }
    for(const auto &line: layer5)
    {
        for(const auto &elem: line)
        {
            fout5 << elem << " "; 
        }
        fout5 << endl; 
    }
    for(const auto &line: layer5)
    {
        for(const auto &elem: line)
        {
            fout5 << elem << " "; 
        }
        fout5 << endl; 
    }
    for(const auto &line: layer6)
    {
        for(const auto &elem: line)
        {
            fout6 << elem << " "; 
        }
        fout6 << endl; 
    }
    for(const auto &line: layer7)
    {
        for(const auto &elem: line)
        {
            fout7 << elem << " "; 
        }
        fout7 << endl; 
    }
    for(const auto &line: layer8)
    {
        for(const auto &elem: line)
        {
            fout8 << elem << " "; 
        }
        fout8 << endl; 
    }
    for(const auto &line: layer9)
    {
        for(const auto &elem: line)
        {
            fout9 << elem << " "; 
        }
        fout9 << endl; 
    }
    for(const auto &line: layer10)
    {
        for(const auto &elem: line)
        {
            fout10 << elem << " "; 
        }
        fout10 << endl; 
    }
    for(const auto &line: layer11)
    {
        for(const auto &elem: line)
        {
            fout11 << elem << " "; 
        }
        fout11 << endl; 
    }
    
    fout0.close(); 
    fout1.close(); 
    fout2.close(); 
    fout3.close(); 
    fout4.close(); 
    fout5.close(); 
    fout6.close(); 
    fout7.close(); 
    fout8.close(); 
    fout9.close(); 
    fout10.close(); 
    fout11.close(); 
    
    return 0; 
}


#endif

