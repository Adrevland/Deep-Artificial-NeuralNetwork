#ifndef NEURALNETWORK_IMPORTER_H
#define NEURALNETWORK_IMPORTER_H

#include <vector>
#include <iostream>
typedef double Scalar;

class Importer {
public:
    static std::vector<std::vector<Scalar>> GetCSVfile(const char* CsvFile, bool BAnsFirst=false);
    static size_t GetOutputCount(std::vector<std::vector<Scalar>> input);
    static size_t GetInputCount(std::vector<std::vector<Scalar>> input);
    static std::vector<std::vector<Scalar>> GetTestData(std::vector<std::vector<Scalar>> input, int size);
    static void PrintMeme(){
        std::cout << "TRANING MACHINE LEARNINGS MODELL!!" << std::endl;
        std::cout << "BIG DATA!!" << std::endl;
        std::cout << "MACHINE LEARNING!!!" << std::endl;
        std::cout << "BLOCK CHAIN!!!" << std::endl;
        std::cout << "ARTIFICIAL INTELIGENSE!!" << std::endl;
        std::cout << "DIGITAL MANIFACTURING!!; " << std::endl;
        std::cout << "BIG DATA ANALYSIS!!! " << std::endl;
        std::cout << "QUANTUM COMMUNICATION!!!" << std::endl;
        std::cout << "AND INTERNET OF THINGS!!!" << std::endl;
    };
private:

};


#endif //NEURALNETWORK_IMPORTER_H
