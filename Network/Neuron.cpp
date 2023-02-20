#include "Neuron.h"
//Sigmoid function 1/(1+e^-x)

Scalar Neuron::Sigmoid(const Scalar &z) {

    return 1.0 / (1.0 + std::exp(-z));
}

Scalar Neuron::DerivateSigmoid(const Scalar &z) {

    return Sigmoid(z) * (1.0 - Sigmoid(z));
}

/*
Tensor Neuron::DerivateSigmoid(const Tensor &z) {

    Tensor output = Tensor(z.rows(), z.cols());

    for (unsigned int m = 0; m < z.rows(); m++) {
        for (unsigned int n = 0; n < z.cols(); n++) {
            output(m,n) = DerivateSigmoid(z(m,n));
        }
    }

    return output;
}
*/
Neuron::Neuron(int weightCount) {

    gen = std::mt19937(rd());
    InitWeights(weightCount);
    WeightCount = weightCount;
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
    if (ActivateFunction == ACTIVATION_FUNCTION::sigmoid) {

        for (int i{0}; i < count; i++) {
            double minvalue = -(1.0 / sqrt(count));
            double Maxvalue = (1.0 / sqrt(count));
            std::uniform_real_distribution<> distr(minvalue, Maxvalue);
            Weights.push_back(distr(gen));
        }
        return;
    }

    for (int i{0}; i < count; i++) {

        std::uniform_int_distribution<> distr(0.0, 1000.0);
        Scalar weight = distr(gen) / 1000.0;
        Weights.push_back(weight);
    }


}

void Neuron::Activate(std::vector<Scalar> inputs) {
    Activation = -Bias;

    for (size_t i{0}; i < WeightCount; i++) {
        Activation += Weights[i] * inputs[i];
    }


}

Scalar Neuron::BinaryStep(const Scalar &z) {

    return z < 0 ? 0 : 1;
}

Scalar Neuron::DerivateBinaryStep(const Scalar &z) {

    return BinaryStep(z) * (1.0 - BinaryStep(z));
}

/*
Tensor Neuron::DerivateBinaryStep(const Tensor &z) {
    Tensor output = Tensor(z.rows(), z.cols());

    for (unsigned int m = 0; m < z.rows(); m++) {
        for (unsigned int n = 0; n < z.cols(); n++) {
            output(m,n) = DerivateBinaryStep(z(m,n));
        }
    }

    return output;
}
*/
void Neuron::transfer() {
    switch (ActivateFunction) {
        case sigmoid: {
            Output = Sigmoid(Activation);
            break;
        }
        case binaryStep: {
            Output = BinaryStep(Activation);
            break;
        }

    }

}

Scalar Neuron::GetDerivative() {
    switch (ActivateFunction) {
        case sigmoid: {
            return DerivateSigmoid(Activation);
        }
        case binaryStep: {
            return DerivateBinaryStep(Activation);
        }
    }
    return 0;
}
