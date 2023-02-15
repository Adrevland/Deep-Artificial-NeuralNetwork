#ifndef NEURALNETWORK_TENSOR_H
#define NEURALNETWORK_TENSOR_H

//https://github.com/maitek/tensor-cpp

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <random>

typedef double Scalar;

class Tensor {

public:
    Tensor(unsigned int rows, unsigned int cols, Scalar* data = nullptr);

    Tensor copy();
    unsigned int cols() const;
    unsigned int rows() const;
    Scalar mean();
    Scalar* data() const;
    Tensor transpose();
    std::string toString();

    // Static methods
    static Tensor matmul(const Tensor &A, const Tensor &B);

    static Tensor zeros(unsigned int rows, unsigned int cols);
    static Tensor ones(unsigned int rows, unsigned int cols);
    static Tensor range(unsigned int rows, unsigned int cols);
    static Tensor rand(unsigned int rows, unsigned int cols);
    static Tensor randn(unsigned int rows, unsigned int cols);

    static Tensor zeros_like(const Tensor &other);
    static Tensor ones_like(const Tensor &other);
    static Tensor range_like(const Tensor &other);
    static Tensor rand_like(const Tensor &other);
    static Tensor randn_like(const Tensor &other);

    // Operators
    Tensor operator+(const Tensor &other);
    Tensor operator+(const Scalar c);
    Tensor operator-(const Tensor &other);
    Tensor operator-(const Scalar c);
    Tensor operator*(const Tensor &other);
    Tensor operator*(const Scalar c);

    Scalar& operator[](unsigned int index);
    Scalar& operator()(unsigned int x, unsigned int y) const;

private:
    // Internal input check
    static void checkEqualSize(const Tensor &A, const Tensor &B);
    static void checkMatMulPossible(const Tensor &A, const Tensor &B);
    static void checkInBounds(const Tensor &A, unsigned int x, unsigned int y);

    Scalar *m_data;
    unsigned int m_rows;
    unsigned int m_cols;

};


Tensor::Tensor(unsigned int rows, unsigned int cols, Scalar* data){
    m_data = new Scalar[rows*cols]();
    if(data){
        std::memcpy(m_data,data,rows*cols*sizeof(Scalar));
    }
    m_cols = cols;
    m_rows = rows;
}

unsigned int Tensor::rows() const{
    return m_rows;
}

unsigned int Tensor::cols() const{
    return m_cols;
}

Scalar* Tensor::data() const{
    return m_data;
}

void Tensor::checkInBounds(const Tensor &A, unsigned int x, unsigned int y){
    if(x >= A.m_rows || y >= A.m_cols)
    {
        std::cout << "Error: Index out of bounds."<< std::endl;
        throw 0;
    }
}

void Tensor::checkEqualSize(const Tensor &A, const Tensor &B){
    if(A.m_rows != B.m_rows || A.m_cols != B.m_cols)
    {
        std::cout << "Error: " << "Tensors must have the same size" << std::endl;
        throw 0;
    }
}

void Tensor::checkMatMulPossible(const Tensor &A, const Tensor &B){
    if(A.m_cols != B.m_rows)
    {
        std::cout << "MatMult size error: A.cols() = " << A.m_cols
                  << " != B.rows() = "<< B.m_rows << " " << std::endl;
        throw 0;
    }
}

Tensor Tensor::copy(){
    Tensor copy = Tensor(m_rows,m_cols,m_data);
    return copy;
}

Tensor Tensor::transpose(){

    Tensor transposed = Tensor(m_cols,m_rows);
    //swap data for rows and columns
    for(int i = 0; i < m_cols*m_rows; i++){
        int j = (i % m_cols)*m_rows + i/m_cols;
        transposed[j] = m_data[i];
    }
    return transposed;
}

/*
    Static methods
*/

Tensor Tensor::matmul(const Tensor &A, const Tensor &B){
    checkMatMulPossible(A,B);
    unsigned int N = A.m_rows;
    unsigned int K = A.m_cols;
    unsigned int M = B.m_cols;

    Tensor C = Tensor(N,M);
    for(int k = 0; k < K; k++){
        for(int i = 0; i < N; i++){
            for(int j = 0; j < M; j++){
                C(i,j) += A(i,k)*B(k,j);
            }
        }
    }
    return C;
}

Tensor Tensor::zeros(unsigned int rows, unsigned int cols){
    Tensor tensor = Tensor(rows,cols);
    for(int i = 0; i < cols*rows; i++){
        tensor[i] = 0;
    }
    return tensor;
}

Tensor Tensor::ones(unsigned int rows, unsigned int cols){
    Tensor tensor = Tensor(rows,cols);
    for(int i = 0; i < cols*rows; i++){
        tensor[i] = 1;
    }
    return tensor;
}

Tensor Tensor::range(unsigned int rows, unsigned int cols){
    Tensor tensor = Tensor(rows,cols);
    for(int i = 0; i < cols*rows; i++){
        tensor[i] = i;
    }
    return tensor;
}

Tensor Tensor::rand(unsigned int rows, unsigned int cols){
    Tensor tensor = Tensor(rows,cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for(int i = 0; i < cols*rows; i++){
        tensor[i] = dis(gen);
    }
    return tensor;
}

Tensor Tensor::randn(unsigned int rows, unsigned int cols){
    Tensor tensor = Tensor(rows,cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, 1);
    for(int i = 0; i < cols*rows; i++){
        tensor[i] = dis(gen);
    }
    return tensor;
}

Tensor Tensor::zeros_like(const Tensor &other){
    return Tensor::zeros(other.rows(),other.cols());
}
Tensor Tensor::ones_like(const Tensor &other){
    return Tensor::ones(other.rows(),other.cols());
}
Tensor Tensor::range_like(const Tensor &other){
    return Tensor::range(other.rows(),other.cols());
}
Tensor Tensor::rand_like(const Tensor &other){
    return Tensor::rand(other.rows(),other.cols());
}

/*
    Operators
*/
Scalar& Tensor::operator[](unsigned int index){
    return m_data[index];
}

Scalar& Tensor::operator()(unsigned int x, unsigned int y) const {
    checkInBounds(*this,x,y);
    return m_data[x*m_cols+y];
}


Tensor Tensor::operator+(const Tensor &other){
    checkEqualSize(*this,other);
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){
        result.m_data[i] += other.m_data[i];
    }
    return result;
}

Tensor Tensor::operator+(const Scalar c){
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){
        result.m_data[i] += c;
    }
    return result;
}

Tensor Tensor::operator-(const Tensor &other){
    checkEqualSize(*this,other);
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){
        result.m_data[i] -= other.m_data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Scalar c){
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){
        result.m_data[i] -= c;
    }
    return result;
}


Tensor Tensor::operator*(const Tensor &other){
    checkEqualSize(*this,other);
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){
        result.m_data[i] *= other.m_data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Scalar c){
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){
        result.m_data[i] *= c;
    }
    return result;
}

std::string Tensor::toString(){
    std::stringstream ss;

    for(int i = 0; i < m_rows; i++){
        for(int j = 0; j < m_cols; j++){
            ss << std::setprecision(2) << m_data[i*m_cols+j] << " ";
        }
        ss << std::endl;
    }
    return ss.str();
}

Scalar Tensor::mean() {
    double sum = 0;
    for (int i=0; i<rows(); i++){
        for (int j=0; j<cols(); j++){
            checkInBounds(*this,i,j);
            sum += m_data[i*m_cols+j];
        }
    }

    return sum/(rows()*cols());
}


#endif //NEURALNETWORK_TENSOR_H
