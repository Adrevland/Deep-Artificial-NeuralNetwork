#include "Layer.h"

Layer::Layer(int neurons, int weights) {
    initNeurons(neurons,weights);
}

Layer::~Layer() {
    for(auto neuron : Neurons){
        delete neuron;
    }
    Neurons.clear();
}

void Layer::initNeurons(int neurons, int weights) {
    for(int i {0}; i < neurons; i++){
        Neurons.emplace_back(new Neuron(weights));
    }
}
