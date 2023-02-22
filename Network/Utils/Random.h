#ifndef NEURALNETWORK_RANDOM_H
#define NEURALNETWORK_RANDOM_H

#include <random>

static std::random_device rd;
static std::mt19937 gen{std::mt19937(rd())};

#endif //NEURALNETWORK_RANDOM_H
