#include "Importer.h"
#include <fstream>
#include <istream>
#include <iostream>
#include <sstream>
#include <set>
#include <random>

std::vector<std::vector<Scalar>> Importer::GetCSVfile(const char *CsvFile) {

    std::vector<std::vector<Scalar>> Output;

    std::ifstream iStream;
   iStream.open(CsvFile, std::ios::in);

    std::vector<std::string> Row;
    std::string line,word;

    if(!iStream){
        std::cout << "no file";
    }

    while(std::getline(iStream, line)){
        Row.clear();
        std::stringstream lineStream(line);

        if(!lineStream){
            std::cout << "no file";
        }

        while(std::getline(lineStream,word, ',')){
            Row.push_back(word);
        }
        std::vector<Scalar> out;
        for(auto& w : Row){
            out.emplace_back(std::stod(w));
        }
        Output.emplace_back(out);
        out.clear();
    }
    iStream.close();
    return Output;
}

size_t Importer::GetOutputCount(std::vector<std::vector<Scalar>> input) {
    std::set <Scalar>set;
    for(auto row: input){
        set.emplace(row[row.size()-1]);
    }
    //return set.size();
    return *std::max_element(set.begin(), set.end())+1;
}

size_t Importer::GetInputCount(std::vector<std::vector<Scalar>> input) {
    if(input.empty()){
        return 0;
    }
    return input[0].size()-1;
}

std::vector<std::vector<Scalar>> Importer::GetTestData(std::vector<std::vector<Scalar>> input, int size) {

    std::random_device rd;
    std::mt19937 gen = std::mt19937(rd());

    std::vector<std::vector<Scalar>> TestData;

    for(int i{0}; i < size; i++){
        std::uniform_int_distribution<> distr(0, input.size()-1);
        int t = distr(gen);
        auto data = input[t];
        //data.pop_back(); //todo commented out to read excepted output in debugger
        TestData.emplace_back(data);
    }
    return TestData;
}
