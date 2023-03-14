#include <iostream>
#include "Network/Utils/Importer.h"
#include "Network/NeuralNetwork.h"
#include "Network/Utils/ActivationFunctions.h"
#include "Network/ClippedDoubleDQN.h"


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
        TrainingData = Importer::GetCSVfile("../DataSets/Seed.csv", false);
        //TrainingData = Importer::GetCSVfile("../DataSets/mnist_test.csv", true);
        //TrainingData = Importer::GetCSVfile("../DataSets/mnist_train.csv", true);
    }
    int outputSize = Importer::GetOutputCount(TrainingData);
    int inputSize = Importer::GetInputCount(TrainingData);

    float rate = 0.01f;
    int epoch = 1000;
    //std::vector<int> HiddenLayout{80,160,100,20};
    //std::vector<int> HiddenLayout{5,5};
    std::vector<int> HiddenLayout{12,8};
    //std::vector<int> HiddenLayout{10};
    //std::vector<int> HiddenLayout{20};
    //std::vector<int> HiddenLayout{64,16};

    NeuralNetwork network;
    network.bLog = true;

    network.HiddenActivation = &ReLU;
    network.HiddenDerivativeActivation = &DerivateReLU;
    network.HiddenLayerWeightInitType = WeightInitializing::HE;
    network.OutputActivation = &Sigmoid;
    network.OutputDerivativeActivation = &DerivateSigmoid;
    network.OutputLayerWeightInitType = WeightInitializing::XAVIER;
    network.MaxMSE = 0.001;

    network.InitNetwork(inputSize,outputSize,HiddenLayout);
    network.Train(TrainingData, rate, epoch, outputSize,true);

    network.PrintNetwork();


    //test Q Network
    ClippedDoubleDQN qActor;
    qActor.InitQNetwork(HiddenLayout,HiddenLayout,inputSize,outputSize);

    std::cout <<"\n\n";
    std::cout << "-----------------------\n"
                 "      test results\n"
                 "-----------------------\n";
    auto TestData = Importer::GetTestData(TrainingData,30);
    for (auto& data : TestData) {
        int prediction = network.Predict(data);
        auto softmax = network.PredictSoftMaxOutput(data);
        std::cout << "\tExpected=" << data.back() << ", Predicted=" << prediction;
        std::cout << "\t\t"<< (data.back() == prediction ? "Correct" : "<------") << std::endl;
    }
    return 0;


}
