#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

//#include "Tensor.h"
#include <vector>
#include "Layer.h"
#include <iostream>
#include <map>
#include <deque>
#include <utility>

//Defaults to HiddenLayer using  LReLU
//Defaults to OutputLayer Using Sigmoid
class NeuralNetwork {
public:

    NeuralNetwork();
    ~NeuralNetwork();

    void InitNetwork(int inputs,int outputs,std::vector<int> hiddenLayout);
    void AddLayer(int neurons, int weights, Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar),WeightInitializing& WeightType);
    std::vector<Scalar> ForwardPropagate(std::vector<Scalar> inputs);
    void BackwardPropagateError(std::vector<Scalar> &expected);
    void UpdateWeights(std::vector<Scalar>& inputs, Scalar rate);
    void Train(std::vector<std::vector<Scalar>>trainingData, Scalar rate, size_t epoch, size_t outputs, bool BNormalizeData = true);
    long Predict(std::vector<Scalar> input);
    void PrintNetwork();

    //DQN
    long EpsilonGreedy(const std::vector<Scalar>& States);
    void TrainDQN(std::vector<Scalar>&States,std::vector<Scalar>&Outputs, Scalar rate);
    Scalar GetQMax(std::vector<Scalar> & states);

    bool bLog{true};
    std::vector<Scalar> PredictSoftMaxOutput(std::vector<Scalar>& input);
    Scalar (*HiddenActivation)(Scalar){nullptr};
    Scalar (*OutputActivation)(Scalar){nullptr};
    Scalar (*HiddenDerivativeActivation)(Scalar){nullptr};
    Scalar (*OutputDerivativeActivation)(Scalar){nullptr};

    Scalar MaxMSE{0.0001};

    WeightInitializing HiddenLayerWeightInitType{WeightInitializing::XAVIER};
    WeightInitializing OutputLayerWeightInitType{WeightInitializing::XAVIER};

    int RandomActionCount{10};
    std::vector<Layer> Layers;
private:
    int RandomActionsDone{0};
    size_t LayerCount{0};


    std::vector<std::vector<Scalar>> NormalizeData(std::vector<std::vector<Scalar>>& data);
    std::vector<Scalar> NormalizeData(std::vector<Scalar>& data);
    double MaxValue = std::numeric_limits<double>().min();
    double MinValue = std::numeric_limits<double>().max();
    bool DataNormalized{false};

    //Q-Stuff
    //https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

    Scalar Epsilon{1.0};

};


#endif //NEURALNETWORK_NEURALNETWORK_H
