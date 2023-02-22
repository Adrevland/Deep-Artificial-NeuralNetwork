#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include "Neuron.h"
#include <vector>

class Layer {
public:
    Layer(int neurons, int weights,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar));
    ~Layer();
    std::vector<Neuron>& GetNeurons(){return Neurons;}
private:
    void initNeurons(int neurons, int weights,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar));
    std::vector<Neuron> Neurons;
};


#endif //NEURALNETWORK_LAYER_H
