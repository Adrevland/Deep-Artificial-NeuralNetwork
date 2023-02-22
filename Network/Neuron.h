    #ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

//#include "Tensor.h"
#include <random>
//to fast change datatype

typedef double Scalar;

class Neuron {
    public:
    explicit Neuron(int WeightCount,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar));
    ~Neuron();

    Scalar GetDerivative();
    void Activate(std::vector<Scalar> inputs);
    void transfer();
    Scalar GetOutput(){transfer();return Output;}
    Scalar GetActivation(){return Activation;}
    Scalar GetDelta(){return Delta;}
    std::vector<Scalar>& GetWeights(){return Weights;}
    void SetDelta(Scalar delta){Delta = delta;}

    Scalar Bias{0};

    Scalar (*ActivationFunc)(Scalar);
    Scalar (*DerActivationFunc)(Scalar);
private:
    size_t WeightCount{0};
    std::vector<Scalar> Weights;
    Scalar Activation{0};
    Scalar Output{0};
    Scalar Delta{0};

    std::random_device rd;
    std::mt19937 gen;
    void InitWeights(int WeightCount);
};


#endif //NEURALNETWORK_NEURON_H
