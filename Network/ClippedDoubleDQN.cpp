#include "ClippedDoubleDQN.h"
#include "Utils/ActivationFunctions.h"
#include "Utils/Random.h"

ClippedDoubleDQN::ClippedDoubleDQN() {
}

ClippedDoubleDQN::~ClippedDoubleDQN() {
}

void ClippedDoubleDQN::InitQNetwork(std::vector<int> &ActorNetworkLayout, std::vector<int> &CriticNetworkLayout, int &inputs, int &actions) {


    ActorNetwork.bLog = false;

    ActorNetwork.HiddenActivation = &ReLU;
    ActorNetwork.HiddenDerivativeActivation = &DerivateReLU;
    ActorNetwork.HiddenLayerWeightInitType = WeightInitializing::HE;

    ActorNetwork.OutputActivation = &Linear;
    ActorNetwork.OutputDerivativeActivation = &DerivateLinear;
    ActorNetwork.OutputLayerWeightInitType = WeightInitializing::XAVIER;

    CriticNetwork.bLog = false;

    CriticNetwork.HiddenActivation = &ReLU;
    CriticNetwork.HiddenDerivativeActivation = &DerivateReLU;
    CriticNetwork.HiddenLayerWeightInitType = WeightInitializing::HE;

    CriticNetwork.OutputActivation = &Linear;
    CriticNetwork.OutputDerivativeActivation = &DerivateLinear;
    CriticNetwork.OutputLayerWeightInitType = WeightInitializing::XAVIER;

    CriticNetwork.InitNetwork(inputs,actions,CriticNetworkLayout);

    ActorNetwork.InitNetwork(inputs,actions,ActorNetworkLayout);

    NetworkIsInitialized = true;
}

void ClippedDoubleDQN::AddMemory(std::vector<Scalar> states, int action, Scalar reward, std::vector<Scalar> nextStates) {

    ExperiencedReplayMemory.emplace_back(states,action,reward,nextStates);
}

long ClippedDoubleDQN::GetAction(std::vector<Scalar> & states) {
    return ActorNetwork.EpsilonGreedy(states);
}

void ClippedDoubleDQN::Learn() {

    //https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/


    //double DQN
    //https://arxiv.org/pdf/1802.09477v3.pdf
    //https://arxiv.org/pdf/1509.06461.pdf

    if(ExperiencedReplayMemory.size() < BatchSize){
        std::cout << "ReplayMemory to small" << std::endl;
        return;
    }

    //learn Training Network
    for(int i{0}; i < BatchSize; i++){

        auto dist =std::uniform_int_distribution<>(0,ExperiencedReplayMemory.size()-1);
        int randint = dist(gen);
        auto memory = ExperiencedReplayMemory[randint];

        //Scalar gamma = (double)randint/(double)ExperiencedReplayMemory.size();
        //Scalar gamma = std::pow(0.9,(ExperiencedReplayMemory.size()-1)-randint);
        Scalar gamma = 1.0;


        double NewQMaxCritic = CriticNetwork.GetQMax(memory.NewStates);
        double NewQMaxActor = ActorNetwork.GetQMax(memory.NewStates);
        double NewQMax = std::min(NewQMaxCritic,NewQMaxActor);

        std::vector<Scalar> Feedback = ActorNetwork.ForwardPropagate(memory.States);
        Feedback[memory.Action] = memory.Reward + gamma * NewQMax; // r + y*min(qmax2, qmax1)
        ActorNetwork.TrainDQN(memory.NewStates,Feedback,LearningRate);

        Feedback = CriticNetwork.ForwardPropagate(memory.States);
        Feedback[memory.Action] = memory.Reward + gamma * NewQMaxActor;
        CriticNetwork.TrainDQN(memory.NewStates,Feedback,LearningRate);
    }

    EpisodeCount ++;
}

void ClippedDoubleDQN::LearnFromAllMemory() {

    //https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

    //learn
    for(int i{0}; i < ExperiencedReplayMemory.size(); i++){

        auto memory = ExperiencedReplayMemory[i];

        Scalar gamma = std::pow(0.9,(ExperiencedReplayMemory.size()-1)-i);
        gamma = 1.0;


        double NewQMaxCritic = CriticNetwork.GetQMax(memory.NewStates);
        double NewQMaxActor = ActorNetwork.GetQMax(memory.NewStates);
        double NewQMax = std::min(NewQMaxCritic,NewQMaxActor);

        std::vector<Scalar> Feedback = ActorNetwork.ForwardPropagate(memory.States);
        Feedback[memory.Action] = memory.Reward + gamma * NewQMax;
        ActorNetwork.TrainDQN(memory.NewStates,Feedback,LearningRate);

        Feedback = CriticNetwork.ForwardPropagate(memory.States);
        Feedback[memory.Action] = memory.Reward + gamma * NewQMaxActor;
        CriticNetwork.TrainDQN(memory.NewStates,Feedback,LearningRate);

    }

    EpisodeCount ++;

    ClearMemory();

}

void ClippedDoubleDQN::ClearMemory() {
    std::cout << std::endl << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "Clearing Replay Memory" << std::endl;
    std::cout << "--------------------------" << std::endl << std::endl;

    ExperiencedReplayMemory.clear();
}

double ClippedDoubleDQN::GetReward() {
    double sum{0};
    for(auto m: ExperiencedReplayMemory)
    {
        sum += m.Reward;
    }
    return sum;
}

void ClippedDoubleDQN::AddMemory(ReplayMemory &memory) {
    if(ExperiencedReplayMemory.size() > MaxMemorySize)
    {
        ExperiencedReplayMemory.erase(
                ExperiencedReplayMemory.begin(),
                ExperiencedReplayMemory.begin() + ExperiencedReplayMemory.size() / 2);
    }
    ExperiencedReplayMemory.emplace_back(memory);
}

