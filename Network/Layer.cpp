#include "Layer.h"

Layer::Layer(int neurons, int weights,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar), WeightInitializing& WeightType) {
    initNeurons(neurons,weights,ActivateFunc,DerActivateFunc,WeightType);
}

Layer::~Layer() {
    Neurons.clear();
}

void Layer::initNeurons(int neurons, int weights,Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar),WeightInitializing& WeightType) {
    for(int i {0}; i < neurons; i++){
        Neurons.emplace_back(weights,ActivateFunc,DerActivateFunc,WeightType);
    }
}
