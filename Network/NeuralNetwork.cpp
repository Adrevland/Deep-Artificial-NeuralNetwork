
#include "NeuralNetwork.h"
#include <algorithm>
#include <iostream>
#include "Utils/Importer.h"
#include "Utils/Random.h"
#include <chrono>
#include <cmath>
#include <random>
//#include "Utils/ActivationFunctions.h"

NeuralNetwork::NeuralNetwork() {

    /*
    OutputActivation = &LeakyReLU;
    OutputDerivativeActivation = &DerivateLeakyReLU;
    HiddenLayerWeightInitType = WeightInitializing::HE;
    HiddenActivation = &Sigmoid;
    HiddenDerivativeActivation = &DerivateSigmoid;
    HiddenLayerWeightInitType = WeightInitializing::XAVIER;
     */
}

NeuralNetwork::~NeuralNetwork() {
    Layers.clear();
}

void NeuralNetwork::InitNetwork(int inputs, int outputs, std::vector<int> hiddenLayout) {

    if (!HiddenDerivativeActivation || !HiddenActivation || !OutputDerivativeActivation || !OutputActivation) {
        std::cout << "Missing one of the activation functions in network" << std::endl;
        return;
    }
    //first hidden connected to input
    AddLayer(hiddenLayout[0], inputs, HiddenActivation, HiddenDerivativeActivation,HiddenLayerWeightInitType);
    //rest of the hidden layers
    if (hiddenLayout.size() >= 2)
        for (int i{1}; i < hiddenLayout.size(); i++) {
            AddLayer(hiddenLayout[i], hiddenLayout[i - 1], HiddenActivation, HiddenDerivativeActivation,HiddenLayerWeightInitType);
        }
    //output layer connected to hidden
    AddLayer(outputs, hiddenLayout.back(), OutputActivation, OutputDerivativeActivation,OutputLayerWeightInitType);
}

void
NeuralNetwork::AddLayer(int neurons, int weights, Scalar (*ActivateFunc)(Scalar), Scalar (*DerActivateFunc)(Scalar), WeightInitializing& WeightType) {
    Layers.emplace_back(neurons, weights, ActivateFunc, DerActivateFunc,WeightType);
    LayerCount++;
}

std::vector<Scalar> NeuralNetwork::ForwardPropagate(std::vector<Scalar> inputs) {
    std::vector<Scalar> NewInputs;

    for (auto &layer: Layers) {
        NewInputs.clear();
        auto& layerNeurons = layer.GetNeurons();
        for (auto& neuron: layerNeurons) {

            neuron.Activate(inputs);

            NewInputs.emplace_back(neuron.GetOutput());

        }
        inputs = NewInputs;
    }

    return NewInputs;
}

void NeuralNetwork::BackwardPropagateError(std::vector<Scalar> expected) {
    //itterate backwards

    //for (int i = Layers.size()-1; i >= 0;i--) {
    for (int B = Layers.size(); B--> 0;) {
        auto& LayerNeurons = Layers[B].GetNeurons();
        for (size_t j{0}; j < LayerNeurons.size(); j++) {
            Scalar error{0.0};
            //output layer
            if (B == Layers.size() - 1) {
                error = expected[j] - LayerNeurons[j].GetOutput();
            }
                //hidden layers
            else {
                for (auto& neuron: Layers[B + 1].GetNeurons()) {
                    error += (neuron.GetWeights()[j] * neuron.GetDelta());
                }
            }
            //update Delta
            LayerNeurons[j].SetDelta(error * LayerNeurons[j].GetDerivative());
        }
    }
}

void NeuralNetwork::UpdateWeights(std::vector<Scalar> &inputs, Scalar rate) {
    for (size_t i{0}; i < Layers.size(); i++) {
        std::vector<Scalar> NewInputs;
        if (i == 0) {
            //original input for first layer (inputlayer)
            NewInputs = std::vector<Scalar>(inputs.begin(), inputs.end());
        } else {
            for (auto &neuron: Layers[i - 1].GetNeurons()) {
                NewInputs.emplace_back(neuron.GetOutput());
            }
        }

        auto& layerNeurons = Layers[i].GetNeurons();
        for (auto& neuron: layerNeurons) {
            auto &weights = neuron.GetWeights();
            for (size_t j = 0; j < weights.size(); j++) {
                //update weights
                weights[j] += rate * neuron.GetDelta() * NewInputs[j];
            }
            //update Bias
            neuron.Bias += rate * neuron.GetDelta();
        }

    }
}

void NeuralNetwork::Train(std::vector<std::vector<Scalar>> trainingData, Scalar rate, size_t epoch, size_t outputs,
                          bool BNormalizeData) {
    if (!HiddenDerivativeActivation || !HiddenActivation || !OutputDerivativeActivation || !OutputActivation) {
        std::cout << "Missing one of the activation functions in network" << std::endl;
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    DataNormalized = BNormalizeData;
    if (bLog)
        Importer::PrintMeme();

    auto normalData = trainingData;
    if (BNormalizeData)
        auto normalData = NormalizeData(trainingData);


    for (size_t i{0}; i < epoch; i++) {
        Scalar MSE{0};
        for (auto &data: normalData) {
            auto out = ForwardPropagate(data);
            //binary exception vector
            std::vector<Scalar> expected(outputs, 0.0);
            expected[static_cast<int>(data.back())] = 1.0;
            //todo map data output to output neurons

            for (size_t j{0}; j < outputs; j++) {
                MSE += std::pow(expected[j] - out[j], 2);
            }
            BackwardPropagateError(expected);
            UpdateWeights(data, rate);
        }
        MSE *= 1.0/(double)(normalData.size());
        if (MSE <= MaxMSE) {
            std::cout << "Breaked out from training in " << i << " epochs" << std::endl;
            std::cout << "Epoch=" << i+1 << ", Rate=" << rate << ", MSE=" << MSE<< std::endl;
            break;
        }
        //print 1% of epochs
        if (bLog) {
            if(epoch>100)
            if((i+1) % (int)(epoch/100) == 0|| i+1 == epoch){
                std::cout << "Epoch=" << i+1 << ", Rate=" << rate << ", MSE=" << MSE << std::endl;
            }

        }

    }
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "\nTraining Time = " << std::chrono::duration<double>(endTime - startTime).count() << std::endl;
}

long NeuralNetwork::Predict(std::vector<Scalar> input) {
    if (!HiddenDerivativeActivation || !HiddenActivation || !OutputDerivativeActivation || !OutputActivation) {
        std::cout << "Missing one of the activation functions in network" << std::endl;
        return 0;
    }

    auto normalInput = input;
    if (DataNormalized)
        normalInput = NormalizeData(input);
    auto outputs = ForwardPropagate(normalInput);
    return std::max_element(outputs.begin(), outputs.end()) - outputs.begin();
}

void NeuralNetwork::PrintNetwork() {
    std::cout << "\n[Neural Network] (Layers: " << Layers.size() << ")" << std::endl;

    std::cout << "{" << std::endl;
    for (size_t l = 0; l < Layers.size(); l++) {
        Layer &layer = Layers[l];
        if (l != Layers.size() - 1)
            std::cout << "\t (Hidden " << l + 1;
        else
            std::cout << "\t (Output  ";
        std::cout << " Neurons: " << Layers[l].GetNeurons().size() << "): {";
        for (size_t i = 0; i < layer.GetNeurons().size(); i++) {
            auto& neuron = layer.GetNeurons()[i];
            std::cout << "\n\t\t\t(Neuron " << i + 1 << "): [ weights " << neuron.GetWeights().size() << " ={";
            auto &weights = neuron.GetWeights();
            for (size_t w = 0; w < weights.size(); ++w) {
                std::cout << weights[w];
                if (w < weights.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << " Bias = " << neuron.Bias << " ]";
        }
        std::cout << "\n\t\t\t}\n\n";
    }
    std::cout << "}" << std::endl;
}

std::vector<std::vector<Scalar>> NeuralNetwork::NormalizeData(std::vector<std::vector<Scalar>> &data) {

    for (auto &line: data) {
        Scalar min = *std::min_element(line.begin(), line.end());
        Scalar max = *std::max_element(line.begin(), line.end());
        if (min < MinValue)
            MinValue = min;
        if (max >= MaxValue)
            MaxValue = max;
    }
    auto normalize = [](double x, double min, double max) {
        return (x - min) / (max - min);
    };
    std::vector<std::vector<Scalar>> output;
    for (auto &line: data) {
        std::vector<Scalar> normalLine;
        for (auto i: line) {
            normalLine.emplace_back(normalize(i, MinValue, MaxValue));
        }
        //copy answer
        normalLine.back() = line.back();
        output.emplace_back(normalLine);
    }

    return output;
}

std::vector<Scalar> NeuralNetwork::NormalizeData(std::vector<Scalar> &data) {
    auto normalize = [](double x, double min, double max) {
        return (x - min) / (max - min);
    };
    std::vector<Scalar> normalLine;
    for (auto i: data) {
        normalLine.emplace_back(normalize(i, MinValue, MaxValue));
    }

    //if input data is larger than output count then copy answer
    if (data.size() > Layers.back().GetNeurons()[0].GetWeights().size()) {
        normalLine.back() = data.back();
    }
    return normalLine;
}

std::vector<Scalar> NeuralNetwork::PredictSoftMaxOutput(std::vector<Scalar> &input) {
    auto normalInput = input;
    if (DataNormalized)
        normalInput = NormalizeData(input);
    auto outputs = ForwardPropagate(normalInput);
    std::vector<Scalar> SoftMaxed = outputs;
    for(int i{0}; i < outputs.size(); i++){
        double probSum{0.0};
        for(int j{0}; j < outputs.size();j++){
            probSum += std::exp(outputs[j]);
        }
        SoftMaxed[i] = std::exp(outputs[i])/probSum;
    }

    return SoftMaxed;
}

std::vector<std::map<std::string, double>>& NeuralNetwork::BuildQTable(int &states, std::vector<std::string> actions) {

    Actions = actions;
    for(int i{0}; i < states; i++){
        std::map<std::string, double> Qtmp;
        for(auto& a : actions){
            Qtmp.insert({a,0.0});
        }
        QTable.emplace_back(Qtmp);
    }

    return QTable;
}

std::string NeuralNetwork::ChooseAction(const int &state) {
    std::uniform_real_distribution<> distr(0.0, 1.0);
    double rand{distr(gen)};
    std::string returnAction;

    //exploit "Get best value from QTable"
    if(rand < Epsilon){
        auto actions = QTable[state];
        double MaxState{0.0};
        for(auto &s: actions){
            if(s.second >= MaxState){
                MaxState = s.second;
                returnAction = s.first;
            }
        }
    }
    else{
        //explore
        std::uniform_int_distribution<> idistr(0,Actions.size()-1);
        int iRand {idistr(gen)};
        returnAction = Actions[iRand];
    }
    return returnAction;
}



