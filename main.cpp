#include <iostream>
#include "Network/NeuralNetwork.h"
#include <set>

int main() {

    std::cout << "TRANING MACHINE LEARNINGS MODELL!!" << std::endl;
    std::cout << "BIG DATA!!" << std::endl;
    std::cout << "MACHINE LEARNING!!!" << std::endl;
    std::cout << "BLOCK CHAIN!!!" << std::endl;
    std::cout << "ARTIFICIAL INTELIGENSE!!" << std::endl;
    std::cout << "DIGITAL MANIFACTURING!!; " << std::endl;
    std::cout << "BIG DATA ANALYSIS!!! " << std::endl;
    std::cout << "QUANTUM COMMUNICATION!!!" << std::endl;
    std::cout << "AND INTERNET OF THINGS!!!" << std::endl;

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

    float rate = 0.4f;
    int epoch = 500;
    int hidden = 2;

    NeuralNetwork network;
    network.bLog = true;
    //network.AddLayer(2,TrainingData.size());
    network.InitNetwork(TrainingData[0].size()-1, hidden, 2);
    network.Train(TrainingData, rate, epoch, 2);


    network.PrintNetwork();

    for (const auto& data : TrainingData) {
        int prediction = network.Predict(data);
        std::cout << "\tExpected=" << data.back() << ", Got=" << prediction << std::endl;
    }
    return 0;


}
