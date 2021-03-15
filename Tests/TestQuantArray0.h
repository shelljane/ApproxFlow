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

int testMNIST(const string lutfile = "../Utils/zli_LUT_ours.txt")
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

int testFashionMNIST(const string &lutfile = "../Utils/zli_LUT_ours.txt")
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
int testCIFAR10(const string &lutfile = "../Utils/zli_LUT_ours.txt")
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


#endif

