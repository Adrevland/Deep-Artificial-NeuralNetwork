#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

//#include "Tensor.h"
#include <vector>
#include "Layer.h"
#include <iostream>

//Defaults to HiddenLayer using  LReLU
//Defaults to OutputLayer Using Sigmoid
class NeuralNetwork {
public:

    NeuralNetwork();
    ~NeuralNetwork();

    void InitNetwork(int inputs,int outputs,std::vector<int> hiddenLayout);
    void AddLayer(int neurons, int weights, Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar),WeightInitializing& WeightType);
    std::vector<Scalar> ForwardPropagate(std::vector<Scalar> inputs);
    void BackwardPropagateError(std::vector<Scalar> expected);
    void UpdateWeights(std::vector<Scalar>& inputs, Scalar rate);
    void Train(std::vector<std::vector<Scalar>>trainingData, Scalar rate, size_t epoch, size_t outputs, bool BNormalizeData = true);
    long Predict(std::vector<Scalar> input);

    void PrintNetwork();
    bool bLog{true};

    Scalar (*HiddenActivation)(Scalar){nullptr};
    Scalar (*OutputActivation)(Scalar){nullptr};
    Scalar (*HiddenDerivativeActivation)(Scalar){nullptr};
    Scalar (*OutputDerivativeActivation)(Scalar){nullptr};

    //max error in %
    Scalar MaxError{1};

    WeightInitializing HiddenLayerWeightInitType{WeightInitializing::XAVIER};
    WeightInitializing OutputLayerWeightInitType{WeightInitializing::XAVIER};

private:
    size_t LayerCount{0};
    std::vector<Layer> Layers;

    std::vector<std::vector<Scalar>> NormalizeData(std::vector<std::vector<Scalar>>& data);
    std::vector<Scalar> NormalizeData(std::vector<Scalar>& data);
    double MaxValue = std::numeric_limits<double>().min();
    double MinValue = std::numeric_limits<double>().max();
    bool DataNormalized{false};
};


#endif //NEURALNETWORK_NEURALNETWORK_H
