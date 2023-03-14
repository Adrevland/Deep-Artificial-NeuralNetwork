
#ifndef NEURALNETWORK_CLIPPEDDOUBLEDQN_H
#define NEURALNETWORK_CLIPPEDDOUBLEDQN_H

#include <vector>
#include "NeuralNetwork.h"

typedef double Scalar;
struct ReplayMemory{
    ReplayMemory() = default;
    ReplayMemory(std::vector<Scalar>& states, int& action, Scalar &reward, std::vector<Scalar>& nextStates){
        States = states;
        Action = action;
        Reward = reward;
        NewStates = nextStates;
    }
    std::vector<Scalar> States;
    int Action{0};
    Scalar Reward{0.0};
    std::vector<Scalar> NewStates;
};

class ClippedDoubleDQN {
public:
    ClippedDoubleDQN();
    ~ClippedDoubleDQN();

    void InitQNetwork(std::vector<int> &ActorNetworkLayout,std::vector<int> &CriticNetworkLayout,int& inputs,int &actions);
    void Learn();
    void LearnFromAllMemory();
    void AddMemory(std::vector<Scalar> states, int action, Scalar reward, std::vector<Scalar> nextStates);
    void AddMemory(ReplayMemory & memory);
    long GetAction(std::vector<Scalar>& states);
    void ClearMemory();
    void SetNetwork(NeuralNetwork nn){ ActorNetwork = nn;}
    NeuralNetwork& GetNetwork(){return ActorNetwork;}
    bool IsInitialized(){return NetworkIsInitialized;}
    double GetReward();
private:

    Scalar LearningRate{0.000025};

    NeuralNetwork ActorNetwork;
    NeuralNetwork CriticNetwork;

    std::vector<ReplayMemory> ExperiencedReplayMemory;

    int EpisodeCount{0};
    int BatchSize{100};
    bool NetworkIsInitialized{false};
    int MaxMemorySize{10000};
};


#endif //NEURALNETWORK_CLIPPEDDOUBLEDQN_H
