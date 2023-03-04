
#ifndef NEURALNETWORK_QACTOR_H
#define NEURALNETWORK_QACTOR_H

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

class QActor {
public:
    QActor();
    ~QActor();

    void InitQNetwork(std::vector<int> &NetworkLayout,int& inputs,int &actions);
    void Learn();
    void LearnFromAllMemory();
    void AddMemory(std::vector<Scalar> states, int action, Scalar reward, std::vector<Scalar> nextStates);
    void AddMemory(ReplayMemory & memory);
    long GetAction(std::vector<Scalar>& states);
    void ClearMemory();
    void SetNetwork(NeuralNetwork nn){ TrainingNetwork = nn;}
    NeuralNetwork& GetNetwork(){return TrainingNetwork;}
    bool IsInitialized(){return NetworkIsInitialized;}
    double GetReward();
private:

    Scalar LearningRate{0.01};

    NeuralNetwork TrainingNetwork;
    NeuralNetwork MainNetwork;

    std::vector<ReplayMemory> ExperiencedReplayMemory;

    int EpisodeCount{0};
    int BatchSize{10};
    bool NetworkIsInitialized{false};
};


#endif //NEURALNETWORK_QACTOR_H
