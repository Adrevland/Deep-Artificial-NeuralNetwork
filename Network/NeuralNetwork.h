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
    std::vector<Scalar> ForwardPropagate(std::vector<Scalar> inputs);
    void BackwardPropagateError(std::vector<Scalar> expected);
    void UpdateWeights(std::vector<Scalar>& inputs, Scalar rate);
    void Train(std::vector<std::vector<Scalar>>trainingData, Scalar rate, size_t epoch, size_t outputs, bool BNormalizeData = true);
    long Predict(std::vector<Scalar> input);

    void PrintNetwork();
    bool bLog{true};
private:
    size_t LayerCount{0};
    std::vector<Layer*> Layers;
    float MaxError{0.05};

    std::vector<std::vector<Scalar>> NormalizeData(std::vector<std::vector<Scalar>>& data);
    std::vector<Scalar> NormalizeData(std::vector<Scalar>& data);
    double MaxValue = std::numeric_limits<double>().min();
    double MinValue = std::numeric_limits<double>().max();
    bool DataNormalized{false};
};


#endif //NEURALNETWORK_NEURALNETWORK_H
