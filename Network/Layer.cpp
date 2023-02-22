#include "Layer.h"

Layer::Layer(int neurons, int weights,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar)) {
    initNeurons(neurons,weights,ActivateFunc,DerActivateFunc);
}

Layer::~Layer() {
    Neurons.clear();
}

void Layer::initNeurons(int neurons, int weights,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar)) {
    for(int i {0}; i < neurons; i++){
        Neurons.emplace_back(weights,ActivateFunc,DerActivateFunc);
    }
}
