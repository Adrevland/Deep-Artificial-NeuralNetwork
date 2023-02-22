#ifndef NEURALNETWORK_ACTIVATIONFUNCTIONS_H
#define NEURALNETWORK_ACTIVATIONFUNCTIONS_H

#include <cmath>

typedef double Scalar;


//Sigmoid function 1/(1+e^-x)

Scalar Sigmoid(Scalar z){
    return 1.0 / (1.0 + std::exp(-z));
}
Scalar DerivateSigmoid(Scalar z){
    return Sigmoid(z) * (1.0 - Sigmoid(z));
}

Scalar BinaryStep(Scalar z){
    return z < 0 ? 0 : 1;
}
Scalar DerivateBinaryStep(Scalar z){
    return 0;
}

//make leaky ReLU



#endif //NEURALNETWORK_ACTIVATIONFUNCTIONS_H
