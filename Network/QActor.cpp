#include "QActor.h"
#include "Utils/ActivationFunctions.h"
#include "Utils/Random.h"

QActor::QActor() {
}

QActor::~QActor() {
}

void QActor::InitQNetwork(std::vector<int> &NetworkLayout, int &inputs, int &actions) {


    TrainingNetwork.bLog = false;

    TrainingNetwork.HiddenActivation = &ReLU;
    TrainingNetwork.HiddenDerivativeActivation = &DerivateReLU;
    TrainingNetwork.HiddenLayerWeightInitType = WeightInitializing::HE;

    TrainingNetwork.OutputActivation = &Linear;
    TrainingNetwork.OutputDerivativeActivation = &DerivateLinear;
    TrainingNetwork.OutputLayerWeightInitType = WeightInitializing::XAVIER;

    TrainingNetwork.InitNetwork(inputs,actions,NetworkLayout);

    MainNetwork = TrainingNetwork;
}

void QActor::AddMemory(std::vector<Scalar> states, int action, Scalar reward, std::vector<Scalar> nextStates) {

    ExperiencedReplayMemory.emplace_back(states,action,reward,nextStates);
}

long QActor::GetAction(std::vector<Scalar> & states) {
    return TrainingNetwork.EpsilonGreedy(states);
}

void QActor::Learn() {

    //https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

    if(ExperiencedReplayMemory.size() < BatchSize){
        std::cout << "ReplayMemory to small" << std::endl;
        return;
    }
    //copy network every 100 episodes
    if(EpisodeCount % 100 == 0 ){
        MainNetwork = TrainingNetwork;
    }

    //learn
    for(int i{0}; i < BatchSize; i++){

        auto dist =std::uniform_int_distribution<>(0,ExperiencedReplayMemory.size()-1);
        int randint = dist(gen);
        auto memory = ExperiencedReplayMemory[randint];

        //Scalar gamma = (double)randint/(double)ExperiencedReplayMemory.size();
        Scalar gamma = std::pow(0.9,(ExperiencedReplayMemory.size()-1)-randint);

        std::vector<Scalar> Feedback = TrainingNetwork.ForwardPropagate(memory.States);

        double NewQMax = TrainingNetwork.GetQMax(memory.NewStates);

        Feedback[memory.Action] = memory.Reward + gamma * NewQMax;

        TrainingNetwork.TrainDQN(memory.NewStates,Feedback,LearningRate);

    }
    EpisodeCount ++;
}

void QActor::LearnFromAllMemory() {

    //https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

    //learn
    for(int i{0}; i < ExperiencedReplayMemory.size(); i++){

        auto memory = ExperiencedReplayMemory[i];

        Scalar gamma = std::pow(0.9,(ExperiencedReplayMemory.size()-1)-i);

        std::vector<Scalar> Feedback = TrainingNetwork.ForwardPropagate(memory.States);

        double NewQMax = TrainingNetwork.GetQMax(memory.NewStates);

        Feedback[memory.Action] = memory.Reward + gamma * NewQMax;

        TrainingNetwork.TrainDQN(memory.NewStates,Feedback,LearningRate);

    }
    EpisodeCount ++;

   ClearMemory();

}

void QActor::ClearMemory() {
    std::cout << std::endl << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "Clearing Replay Memory" << std::endl;
    std::cout << "--------------------------" << std::endl << std::endl;

    ExperiencedReplayMemory.clear();
}

