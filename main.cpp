#include <iostream>
#include "Network/Utils/Importer.h"
#include "Network/NeuralNetwork.h"
#include <set>

int main() {



    std::vector<std::vector<Scalar>> TrainingData {
            {2.7810836,		2.550537003,	0},
            {1.465489372,	2.362125076,	0},
            {3.396561688,	4.400293529,	0},
            {1.38807019,	1.850220317,	0},
            {3.06407232,	3.005305973,	0},
            {7.627531214,	2.759262235,	1},
            {5.332441248,	2.088626775,	1},
            {6.922596716,	1.77106367,		1},
            {8.675418651,	-0.242068655,	1},
            {7.673756466,	3.508563011,	1}
    };

    const bool AdvancedData{true};
    if(AdvancedData){
        //TrainingData = Importer::GetCSVfile("../DataSets/Seed.csv", false);
        TrainingData = Importer::GetCSVfile("../DataSets/mnist_test.csv", true);
    }
    int outputSize = Importer::GetOutputCount(TrainingData);
    int inputSize = Importer::GetInputCount(TrainingData);

    float rate = 0.08f;
    int epoch = 500;
    int hidden = 10;
    std::vector<int> HiddenLayout{20};

    NeuralNetwork network;
    network.bLog = true;

    //network.InitNetwork(inputSize, outputSize,hidden );
    network.InitNetwork(inputSize,outputSize,HiddenLayout);
    network.Train(TrainingData, rate, epoch, outputSize,true);


    network.PrintNetwork();

    auto TestData = Importer::GetTestData(TrainingData,10);
    for (const auto& data : TestData) {
        int prediction = network.Predict(data);
        std::cout << "\tExpected=" << data.back() << ", Got=" << prediction << std::endl;
    }
    return 0;


}
