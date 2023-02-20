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

    void InitNetwork(int inputs,int outputs, int hidden);
    void InitNetwork(int inputs,int outputs,std::vector<int> hiddenLayout);
    void AddLayer(int neurons, int weights);
    std::vector<Scalar> ForwardPropagate(std::vector<Scalar>& inputs);
    void BackwardPropagateError(std::vector<Scalar> expected);
    void UpdateWeights(std::vector<Scalar>& inputs, Scalar rate);
    void Train(std::vector<std::vector<Scalar>>trainingData, Scalar rate, size_t epoch, size_t outputs);
    long Predict(std::vector<Scalar> input);

    void PrintNetwork();
    bool bLog{true};
private:
    size_t LayerCount{0};
    std::vector<Layer*> Layers;
    float MaxError{0.05};
};


#endif //NEURALNETWORK_NEURALNETWORK_H
