#include "Neuron.h"
//#include "Utils/ActivationFunctions.h"
#include "Utils/Random.h"

Neuron::Neuron(int weightCount, Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar), WeightInitializing& WeightType) {

    InitWeights(weightCount,WeightType);
    //WeightCount = weightCount;

    ActivationFunc = ActivateFunc;
    DerActivationFunc = DerActivateFunc;
}

Neuron::~Neuron() {

}

void Neuron::InitWeights(int count, WeightInitializing& WeightType) {

    //set by Stanford university bias
    //https://cs231n.github.io/neural-networks-2/
    Bias = 0.01;


    //xavier weights for sigmoid and tanh
    //https://cs230.stanford.edu/section/4/

    //HE weights for ReLU
    //weight = G (0.0, sqrt(2/n))
    double minValue{0};
    double maxValue{0};
    switch (WeightType) {
        case XAVIER : {
            minValue = -(1.0 / sqrt(count));
            maxValue = (1.0 / sqrt(count));
            break;
        }
        case HE : {
            minValue = 0.0;
            maxValue = sqrt(2.0/count);
            break;
        }
    }

    for (int i{0}; i < count; i++) {
        std::uniform_real_distribution<> distr(minValue, maxValue);
        Weights.push_back(distr(gen));
    }

}

void Neuron::Activate(const std::vector<Scalar>& inputs) {
    Activation = +Bias;

    for (size_t i{0}; i < Weights.size(); i++) {
        Activation += Weights[i] * inputs[i];
    }


}

void Neuron::transfer() {

    Output = ActivationFunc(Activation);

}

Scalar Neuron::GetDerivative() {

    return DerActivationFunc(Activation);

}
