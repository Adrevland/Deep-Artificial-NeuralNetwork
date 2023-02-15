cmake_minimum_required(VERSION 3.24)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork main.cpp Tensor.h Neuron.cpp Neuron.h NeuralNetwork.cpp NeuralNetwork.h Layer.cpp Layer.h)
