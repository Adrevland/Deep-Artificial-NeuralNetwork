#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

//#include "Tensor.h"
#include <vector>
#include "Layer.h"
#include <iostream>

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void InitNetwork(int inputs, int hidden,int outputs);
    void AddLayer(int neurons, int weights);
    std::vector<Scalar> ForwardPropagate(std::vector<Scalar> inputs);
    void BackwardPropagateError(std::vector<Scalar> expected);
    void UpdateWeights(std::vector<Scalar> inputs, Scalar rate);
    void Train(std::vector<std::vector<Scalar>>trainingData, Scalar rate, size_t epoch, size_t outputs);
    Scalar Predict(std::vector<Scalar> input);

    void PrintNetwork();
    bool bLog{true};
private:
    void PrintMeme(){
        std::cout << "TRANING MACHINE LEARNINGS MODELL!!" << std::endl;
        std::cout << "BIG DATA!!" << std::endl;
        std::cout << "MACHINE LEARNING!!!" << std::endl;
        std::cout << "BLOCK CHAIN!!!" << std::endl;
        std::cout << "ARTIFICIAL INTELIGENSE!!" << std::endl;
        std::cout << "DIGITAL MANIFACTURING!!; " << std::endl;
        std::cout << "BIG DATA ANALYSIS!!! " << std::endl;
        std::cout << "QUANTUM COMMUNICATION!!!" << std::endl;
        std::cout << "AND INTERNET OF THINGS!!!" << std::endl;
    };
    size_t LayerCount{0};
    std::vector<Layer*> Layers;
};


#endif //NEURALNETWORK_NEURALNETWORK_H
