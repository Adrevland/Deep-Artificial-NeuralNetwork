#include <iostream>
#include "NeuralNetwork.h"
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


    std::vector<std::vector<Scalar>> TrainingData{
            {0, 0, 1},
            {1, 0, 0},
            {0, 1, 0},
            {1, 1, 1},
            {0, 0, 1},
            {1, 0, 0},
            {0, 1, 0},
            {1, 1, 1},
            {0, 0, 1},
            {1, 0, 0},
            {0, 1, 0},
            {1, 1, 1},
    };

    float rate = 0.1f;    // how much of an impact shall an error have on a weight
    int epoch = 500;        // how many times should weights be updated
    int hidden = 1;        // how many neurons you want in the first layer

    NeuralNetwork network;
    network.bLog = true;
    //network.AddLayer(2,TrainingData.size());
    network.InitNetwork(TrainingData[0].size()-1, hidden, 2);
    network.Train(TrainingData, rate, epoch, 2);


    network.PrintNetwork();

    std::vector<Scalar> test{0, 0};
    std::cout << "Testing \n 0,0 : " << network.Predict(test) << std::endl;
    std::vector<Scalar> test2{0, 1};
    std::cout << "Testing \n 0,1 : " << network.Predict(test2) << std::endl;
    std::vector<Scalar> test3{1, 0};
    std::cout << "Testing \n 1,0 : " << network.Predict(test3) << std::endl;
    std::vector<Scalar> test4{1, 1};
    std::cout << "Testing \n 1,1 : " << network.Predict(test4) << std::endl;
    return 0;


}
