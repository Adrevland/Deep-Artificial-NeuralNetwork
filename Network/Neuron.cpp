#include "Neuron.h"
//#include "Utils/ActivationFunctions.h"
#include "Utils/Random.h"

Neuron::Neuron(int weightCount, Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar)) {

    InitWeights(weightCount);
    //WeightCount = weightCount;

    ActivationFunc = ActivateFunc;
    DerActivationFunc = DerActivateFunc;
}

Neuron::~Neuron() {

}

void Neuron::InitWeights(int count) {



    //set by Stanford university bias
    //https://cs231n.github.io/neural-networks-2/
    Bias = 0.01;


    //xavier weights for sigmoid and tanh
    //https://cs230.stanford.edu/section/4/

    //HE weights for ReLU
    //weight = G (0.0, sqrt(2/n))

    //todo only use for sigmoid and tanh
    for (int i{0}; i < count; i++) {
        double minvalue = -(1.0 / sqrt(count));
        double Maxvalue = (1.0 / sqrt(count));
        std::uniform_real_distribution<> distr(minvalue, Maxvalue);
        Weights.push_back(distr(gen));
    }
    //todo use HE weights for ReLU variants
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
