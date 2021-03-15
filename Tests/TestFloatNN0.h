#include "../Headers/Common.h"
#include "../Headers/Timer.h"
#include "../Headers/Range.h"
#include "../Headers/Ref.h"
#include "../Headers/Thread.h"
#include "../Headers/NDArrayFloatCPU.h"
#include "../Headers/Operator.h"
#include "../Headers/OpsFloat.h"
#include "../Headers/Variable.h"

using namespace std; 
using namespace ApproxFlow; 

constexpr size_t SIZE = 18; 
constexpr size_t HALFSIZE = SIZE/2-1; 

typedef NDArrayFloatCPU<float> NDArray; 
typedef Variable<NDArray> Var; 

vector<vector<float>> getImages()
{
    vector<vector<float>> data; 
    float tmpNums[4]; 
    ifstream fin("../Data/MNIST_TestData.dat"); 
    if(!fin)
    {
        cerr << "ERROR when reading files. " << endl; 
        exit(1); 
    }
    
    fin >> tmpNums[0] >> tmpNums[1] >> tmpNums[2] >> tmpNums[3]; 
//     cout << "Magic Numbers: " << tmpNums[0] << " " << tmpNums[1] << " " << tmpNums[2] << " " << tmpNums[3] << endl; 
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

int testFloatNN0()
{
//     float::InitLUT(); 
    
    vector<string> nameLayers = {
        "Conv1", "Conv2", "Conv3", 
        "FC1", "FC_Logits"
    }; 
    
    map<string, vector<vector<size_t>>> sizeLayers; 
    sizeLayers[nameLayers[0]]  = {{16}, {5 * 5 * 1, 16}};     // Conv1
    sizeLayers[nameLayers[1]]  = {{32}, {3 * 3 * 16, 32}};     // Conv2
    sizeLayers[nameLayers[2]]  = {{64}, {3 * 3 * 32, 64}};     // Conv3
    sizeLayers[nameLayers[3]]  = {{1, 256}, {1024, 256}};    // FC1
    sizeLayers[nameLayers[4]] = {{1, 10}, {256, 10}};        // FC_Logits
    
    // Reading Variables
    ifstream fin("../Weights/MNIST_quantized.txt"); 
    ifstream flog("../Weights/MNIST_quantized.log"); 
    map<string, vector<vector<float>>> mapWeights; 
    for(unsigned idx = 0; idx < nameLayers.size(); idx++)
    {
        vector<vector<float>> listWeights; 
        vector<vector<float>> listWeightsPost; 
        if(nameLayers[idx].substr(0, 2) == "FC")
        {
            cout << "Fully Connected Layer" << endl; 
            unsigned sizeBias, sizeWeights;
            string strTmp; 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeBias = atoi(strTmp.c_str()); 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeWeights = atoi(strTmp.c_str()); 
            cout << sizeBias << " " << sizeWeights << endl; 
            vector<float> vecBias(sizeBias), vecWeights(sizeWeights); 
            for(unsigned jdx = 0; jdx < sizeBias; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecBias[jdx] = floatTmp; 
            }
            for(unsigned jdx = 0; jdx < sizeWeights; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecWeights[jdx] = floatTmp; 
            }
            listWeights.push_back(vecBias); 
            listWeights.push_back(vecWeights); 
        }
        else if(nameLayers[idx].substr(0, 4) == "Conv")
        {
            cout << "Convolution Layer" << endl; 
            unsigned sizeBias, sizeWeights;
            string strTmp; 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeBias = atoi(strTmp.c_str()); 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeWeights = atoi(strTmp.c_str()); 
            cout << sizeBias << " " << sizeWeights << endl; 
            vector<float> vecBias(sizeBias), vecWeights(sizeWeights); 
            for(unsigned jdx = 0; jdx < sizeBias; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecBias[jdx] = floatTmp; 
            }
            for(unsigned jdx = 0; jdx < sizeWeights; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecWeights[jdx] = floatTmp; 
            }
            listWeights.push_back(vecBias); 
            listWeights.push_back(vecWeights); 
        }
        else if(nameLayers[idx].substr(0, 9) == "BatchNorm")
        {
            cout << "Batch Normalization Layer" << endl; 
            unsigned sizeMean, sizeVar, sizeOffset, sizeScale;
            string strTmp; 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeMean = atoi(strTmp.c_str()); 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeVar = atoi(strTmp.c_str()); 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeOffset = atoi(strTmp.c_str()); 
            getline(flog, strTmp); strTmp = ""; getline(flog, strTmp); strTmp = ""; 
            getline(flog, strTmp); 
            sizeScale = atoi(strTmp.c_str()); 
            cout << sizeMean << " " << sizeVar << " " << sizeOffset << " " << sizeScale << endl; 
            vector<float> vecMean(sizeMean), vecVar(sizeVar), vecOffset(sizeOffset), vecScale(sizeScale); 
            for(unsigned jdx = 0; jdx < sizeMean; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecMean[jdx] = floatTmp; 
            }
            for(unsigned jdx = 0; jdx < sizeVar; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecVar[jdx] = floatTmp; 
            }
            for(unsigned jdx = 0; jdx < sizeOffset; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecOffset[jdx] = floatTmp; 
            }
            for(unsigned jdx = 0; jdx < sizeScale; jdx++)
            {
                float floatTmp; 
                fin >> floatTmp; 
                vecScale[jdx] = floatTmp; 
            }
            listWeights.push_back(vecMean);
            listWeights.push_back(vecVar);
            listWeights.push_back(vecOffset);
            listWeights.push_back(vecScale);
        }
        else
        {
            cerr << "Unsupported layer!" << endl; 
        }
        
        float maxAbs = 0.0; 
        for(unsigned jdx = 0; jdx < listWeights.size(); jdx++)
        {
            // Find max. 
            for(size_t kdx = 0; kdx < listWeights[jdx].size(); kdx++)
            {
                if(abs(listWeights[jdx][kdx]) > maxAbs)
                {
                    maxAbs = abs(listWeights[jdx][kdx]); 
                }
            }
        }
        for(unsigned jdx = 0; jdx < listWeights.size(); jdx++)
        {
//             for(size_t kdx = 0; kdx < listWeights[jdx].size(); kdx++)
//             {
//                 listWeights[jdx][kdx] = listWeights[jdx][kdx] / maxAbs;
//             }
            
            listWeightsPost.push_back(vector<float>(listWeights[jdx].size())); 
            for(size_t kdx = 0; kdx < listWeights[jdx].size(); kdx++)
            {
                listWeightsPost[jdx][kdx] = listWeights[jdx][kdx]; 
            }
        }
        cout << "Layer No. " << idx << " MaxVal = " << maxAbs << endl; 
        
        mapWeights[nameLayers[idx]] = listWeightsPost; 
    }
    fin.close(); 
    flog.close(); 
    
    vector<vector<float>> images = getImages(); 
    vector<unsigned> labels = getLabels(); 
    
    Ref<NDArray> image = new NDArray(28, 28, 1); 
    
    Ref<Var> input = new Var(image); 
    
    cout << "Creating Conv1. " << endl; 
    
    Ref<Var> weightConv1 = new Var(new NDArray(sizeLayers["Conv1"][1], mapWeights["Conv1"][1])); 
    Ref<Var> biasConv1   = new Var(new NDArray(sizeLayers["Conv1"][0], mapWeights["Conv1"][0])); 
    Ref<Var> conv1       = Conv2D<NDArray>(input, weightConv1, {5, 5}, {2, 2}, true); 
    Ref<Var> added1      = AddBias<NDArray>(conv1, biasConv1); 
    Ref<Var> acted1      = ReLU<NDArray>(added1); 
    
    cout << "Creating Conv2. " << endl; 
    
    Ref<Var> weightConv2 = new Var(new NDArray(sizeLayers["Conv2"][1], mapWeights["Conv2"][1])); 
    Ref<Var> biasConv2   = new Var(new NDArray(sizeLayers["Conv2"][0], mapWeights["Conv2"][0])); 
    Ref<Var> conv2       = Conv2D<NDArray>(acted1, weightConv2, {3, 3}, {2, 2}, true); 
    Ref<Var> added2      = AddBias<NDArray>(conv2, biasConv2); 
    Ref<Var> acted2      = ReLU<NDArray>(added2); 
    
    cout << "Creating Conv3. " << endl; 
    
    Ref<Var> weightConv3 = new Var(new NDArray(sizeLayers["Conv3"][1], mapWeights["Conv3"][1])); 
    Ref<Var> biasConv3   = new Var(new NDArray(sizeLayers["Conv3"][0], mapWeights["Conv3"][0])); 
    Ref<Var> conv3       = Conv2D<NDArray>(acted2, weightConv3, {3, 3}, {2, 2}, true); 
    Ref<Var> added3      = AddBias<NDArray>(conv3, biasConv3); 
    Ref<Var> acted3      = ReLU<NDArray>(added3); 
    
    cout << "Creating Flatten. " << endl; 
    
    Ref<Var> flattened   = FlattenAll<NDArray>(acted3); 
    
    cout << "Creating FC1. " << endl; 
    
    Ref<Var> weightFC1   = new Var(new NDArray(sizeLayers["FC1"][1], mapWeights["FC1"][1])); 
    Ref<Var> biasFC1     = new Var(new NDArray(sizeLayers["FC1"][0], mapWeights["FC1"][0])); 
    Ref<Var> fc1         = MatMul<NDArray>(flattened, weightFC1); 
    Ref<Var> addedfc1    = Add<NDArray>(fc1, biasFC1); 
    Ref<Var> actedfc1    = ReLU<NDArray>(addedfc1); 
    
    cout << "Creating FC2. " << endl; 
    
    Ref<Var> weightFC2   = new Var(new NDArray(sizeLayers["FC_Logits"][1], mapWeights["FC_Logits"][1])); 
    Ref<Var> biasFC2     = new Var(new NDArray(sizeLayers["FC_Logits"][0], mapWeights["FC_Logits"][0])); 
    Ref<Var> fc2         = MatMul<NDArray>(actedfc1, weightFC2); 
    Ref<Var> addedfc2    = Add<NDArray>(fc2, biasFC2); 
    
    unsigned globalStep = 1; 
    constexpr unsigned TotalStep = 10000; 
    unsigned count = 0; 
    for(unsigned idx = 0; idx < TotalStep; idx++, globalStep++)
    {
        *image = NDArray({28, 28, 1}, images[idx]); 
        addedfc2->evaluate(globalStep); 
        size_t label = addedfc2->value().posmax(); 
        if(label == labels[idx])
        {
            count++; 
        }
//         print(addedfc2->value().shape()); 
//         input->value().print();  
//         added1->value().print();  
//         acted1->value().print();  
//         added2->value().print();  
//         acted2->value().print();  
//         added3->value().print();  
//         acted3->value().print(); 
        addedfc2->value().print(); 
        cout << "Sample No." << idx << " " << "; Label: " << labels[idx] << ", Predicted: " << label << " -> " << ((label == labels[idx]) ? "Right" : "Wrong") << endl; 
    }
    
    cout << "Accuracy: " << static_cast<float>(count) / globalStep << endl; 
    
    return 0; 
}
