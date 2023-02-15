
#include "NeuralNetwork.h"
#include <algorithm>
#include <iostream>

NeuralNetwork::NeuralNetwork() {

}

NeuralNetwork::~NeuralNetwork() {
    for(auto layer : Layers){
        delete layer;
    }
    Layers.clear();
}

void NeuralNetwork::InitNetwork(int inputs, int hidden, int outputs) {
    //hidden layer connected to input
    AddLayer(hidden, inputs+1);
    //output layer connected to hidden
    AddLayer(outputs, hidden+1);
}

void NeuralNetwork::AddLayer(int neurons, int weights) {
    Layers.emplace_back(new Layer(neurons, weights));
    LayerCount++;
}

std::vector<Scalar> NeuralNetwork::ForwardPropagate(std::vector<Scalar> inputs) {
    std::vector<Scalar> NewInputs;
    for (auto layer: Layers) {
        NewInputs.clear();
        auto layerNeurons = layer->GetNeurons();
        for (auto &neuron: layerNeurons) {
            neuron->Activate(inputs);
            NewInputs.emplace_back(neuron->GetOutput());
        }
    }

    return NewInputs;
}

void NeuralNetwork::BackwardPropagateError(std::vector<Scalar> expected) {
    //itterate backwards
    for (size_t i{Layers.size()}; i-- > 0;) {
        auto LayerNeurons = Layers[i]->GetNeurons();
        for (size_t j{0}; j < LayerNeurons.size(); j++) {
            Scalar error{0.0};

            //output layer
            if (i == Layers.size() - 1) {
                error = expected[j] - LayerNeurons[j]->GetOutput();
            }
            //hidden layers
            else {
                for (auto &neuron: Layers[i + 1]->GetNeurons()) {
                    error += (neuron->GetWeights()[j] * neuron->GetDelta());
                }
            }
            //update Delta
            LayerNeurons[j]->SetDelta(error * LayerNeurons[j]->GetDerivative());
        }
    }
}

void NeuralNetwork::UpdateWeights(std::vector<Scalar> inputs, Scalar rate) {
    for(size_t i {0}; i < Layers.size(); i++){
        std::vector<Scalar> NewInputs;
        if(i == 0){
            //original input for first layer (inputlayer)
            NewInputs = std::vector<Scalar>(inputs.begin(),inputs.end());
        }else{
            for(auto neuron: Layers[i-1]->GetNeurons()){
                NewInputs.emplace_back(neuron->GetOutput());
            }
        }

        auto layerNeurons = Layers[i]->GetNeurons();
        for(auto& neuron : layerNeurons){
            auto& weights = neuron->GetWeights();
            for(size_t j = 0; j < NewInputs.size()-1; j++){
                //update weights
                weights[j] += rate*neuron->GetDelta()*NewInputs[j];
            }
            //update Bias
            neuron->Bias += rate*neuron->GetDelta();
        }

    }
}

void NeuralNetwork::Train(std::vector<std::vector<Scalar>> trainingData, Scalar rate, size_t epoch, size_t outputs) {
    if(bLog)
        PrintMeme();
    for(size_t i{0}; i < epoch; i++){
        Scalar errorSum{0};
        for(auto& data: trainingData){
            auto out = ForwardPropagate(data);
            std::vector<Scalar> excepted(outputs,0.0);
            excepted[static_cast<int>(data.back())] = 1.0;
            for(size_t j{0}; j < outputs; j++){
                errorSum += std::pow(excepted[j]-out[j],2);
                //errorSum +=excepted[j]-out[j];
            }
            BackwardPropagateError(excepted);
            UpdateWeights(data, rate);
        }
        if(bLog){
            std::cout << "Epoch=" << i << ", Rate=" << rate << ", Error=" << errorSum << std::endl;
        }

    }
}

Scalar NeuralNetwork::Predict(std::vector<Scalar> input) {
    auto outputs = ForwardPropagate(input);
    return std::max_element(outputs.begin(),outputs.end()) - outputs.begin();
}

void NeuralNetwork::PrintNetwork() {
    std::cout << "[Network] (Layers: " << Layers.size() << ")" << std::endl;

    std::cout << "{" << std::endl;
    for (size_t l = 0; l < Layers.size(); l++)
    {
        Layer* layer = Layers[l];
        std::cout << "\t (Layer " << l << "): {";
        for (size_t i = 0; i < layer->GetNeurons().size(); i++)
        {
            auto neuron = layer->GetNeurons()[i];
            std::cout << "<(Neuron " << i << "): [ weights={";
            auto& weights = neuron->GetWeights();
            for (size_t w = 0; w < weights.size(); ++w)
            {
                std::cout << weights[w];
                if (w < weights.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << " Bias = " << neuron->Bias << " ";
            std::cout << "}, output=" << neuron->GetOutput() << ", activation=" << neuron->GetActivation() << ", delta=" << neuron->GetDelta();
            std::cout << "]>";
            if (i < layer->GetNeurons().size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}";
        if (l < Layers.size() - 1) {
            std::cout << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "}" << std::endl;
}
