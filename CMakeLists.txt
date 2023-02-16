cmake_minimum_required(VERSION 3.24)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork main.cpp Network/Tensor.h Network/Neuron.cpp Network/Neuron.h Network/NeuralNetwork.cpp Network/NeuralNetwork.h Network/Layer.cpp Network/Layer.h Network/Utils/Importer.cpp Network/Utils/Importer.h)
