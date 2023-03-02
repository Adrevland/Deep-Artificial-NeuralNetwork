#ifndef NEURALNETWORK_ACTIVATIONFUNCTIONS_H
#define NEURALNETWORK_ACTIVATIONFUNCTIONS_H

#include <cmath>

typedef double Scalar;
//Sigmoid function 1/(1+e^-x)
static Scalar Sigmoid(Scalar z){
    return 1.0 / (1.0 + std::exp(-z));
}
static Scalar DerivateSigmoid(Scalar z){
    return Sigmoid(z) * (1.0 - Sigmoid(z));
}
static Scalar TanH(Scalar z){
    return (2.0/(1+std::exp(-2.0*z)))-1.0;
}
static Scalar DerivateTanH(Scalar z){
    return 1.0-std::pow(TanH(z),2);
}
static Scalar ReLU(Scalar z){
    return  z < 0 ? 0 : z;
}
static Scalar DerivateReLU(Scalar z){
    return z < 0 ? 0 : 1;
}

static Scalar LeakyReLU(Scalar z){
    return  z < 0 ? z*0.01 : z;
}
static Scalar DerivateLeakyReLU(Scalar z){
    return z < 0 ? 0.01 : 1;
}
static Scalar Linear(Scalar z){
    return z;
}
static Scalar DerivateLinear(Scalar z){
    return 1;
}
#endif //NEURALNETWORK_ACTIVATIONFUNCTIONS_H
