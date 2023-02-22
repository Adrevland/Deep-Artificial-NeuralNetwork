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
Scalar TanH(Scalar z){
    return (2.0/(1+std::exp(-2.0*z)))-1.0;
}
Scalar DerivateTanH(Scalar z){
    return 1.0-std::pow(TanH(z),2);
}
Scalar BinaryStep(Scalar z){
    return z < 0 ? 0 : 1;
}
Scalar DerivateBinaryStep(Scalar z){
    return 0;
}

//todo fix ReLU. 100% wrong when using
Scalar ReLU(Scalar z){
    return  z < 0 ? 0 : z;
}
Scalar DerivateReLU(Scalar z){
    return z < 0 ? 0 : 1;
}

Scalar LeakyReLU(Scalar z){
    return  z < 0 ? z*0.01 : z;
}
Scalar DerivateLeakyReLU(Scalar z){
    return z < 0 ? 0.01 : 1;
}
#endif //NEURALNETWORK_ACTIVATIONFUNCTIONS_H
