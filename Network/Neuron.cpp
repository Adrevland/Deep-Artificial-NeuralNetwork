#include "Neuron.h"
//#include "Utils/ActivationFunctions.h"
#include "Utils/Random.h"

Neuron::Neuron(int weightCount,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar)) {

    InitWeights(weightCount);
    WeightCount = weightCount;

    ActivationFunc = ActivateFunc;
    DerActivationFunc = DerActivateFunc;
}

Neuron::~Neuron() {

}

void Neuron::InitWeights(int count) {

    //set bias
    {
        std::uniform_int_distribution<> distr(0, 1000);
        Scalar weight = distr(gen) / 1000.0;
        Bias = weight;
    }

    //xavier weights for sigmoid and tanh // todo add tanh
    //https://cs230.stanford.edu/section/4/

        for (int i{0}; i < count; i++) {
            double minvalue = -(1.0 / sqrt(count));
            double Maxvalue = (1.0 / sqrt(count));
            std::uniform_real_distribution<> distr(minvalue, Maxvalue);
            Weights.push_back(distr(gen));
        }
}

void Neuron::Activate(std::vector<Scalar> inputs) {
    Activation = -Bias;

    for (size_t i{0}; i < WeightCount; i++) {
        Activation += Weights[i] * inputs[i];
    }


}

void Neuron::transfer() {

    Output = ActivationFunc(Activation);

}

Scalar Neuron::GetDerivative() {

    return DerActivationFunc(Activation);

}
