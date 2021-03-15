#include "./Headers/Common.h"
// #include "./Tests/TestFloatNN0.h"
#include "./Tests/TestQuantArray0.h"

using namespace ApproxFlow; 
using namespace std; 

int main(int argc, char **argv) {
//     testMNIST("../Utils/LUT_ours.txt"); 
//     testFashionMNIST("../Utils/LUT_ours.txt"); 
    testCIFAR10("../Utils/LUT_ours.txt"); 
    
    return 0;
}
