#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include <vector>
typedef double Scalar;

enum WeightInitializing{
    //xavier weights for sigmoid and tanh
    //https://cs230.stanford.edu/section/4/
    XAVIER,
    //HE weights for ReLU kinds
    //https://cs231n.github.io/neural-networks-2/
    HE
};

class Neuron {
    public:
    explicit Neuron(int WeightCount,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar),WeightInitializing& WeightType);
    ~Neuron();

    Scalar GetDerivative();
    void Activate(const std::vector<Scalar>& inputs);
    void transfer();
    Scalar& GetOutput(){transfer();return Output;}
    Scalar& GetActivation(){return Activation;}
    Scalar& GetDelta(){return Delta;}
    std::vector<Scalar>& GetWeights(){return Weights;}
    void SetDelta(Scalar delta){Delta = delta;}

    Scalar Bias{0};

    Scalar (*ActivationFunc)(Scalar);
    Scalar (*DerActivationFunc)(Scalar);
private:
    //size_t WeightCount{0};
    std::vector<Scalar> Weights;
    Scalar Activation{0};
    Scalar Output{0};
    Scalar Delta{0};

    void InitWeights(int WeightCount, WeightInitializing& WeightType);
};


#endif //NEURALNETWORK_NEURON_H
