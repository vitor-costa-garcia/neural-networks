#include <vector>
#include <iostream>
#include <Eigen/Dense>
using Eigen::VectorXf;
using Eigen::MatrixXf;

#define learning_rate 0.0001

//ReLU Layer class - this will be my hidden layers
class ReLU_Layer{
public:
	MatrixXf weights;
	VectorXf bias;
	VectorXf z;
	MatrixXf dL_dW;

	int ni; // Number of neurons inside previous layer
	int no; // Number of neurons inside layer

	ReLU_Layer(int ni, int no) : ni(ni), no(no){
		weights = MatrixXf::Random(no, ni);
		bias = VectorXf::Zero(no);
		z = VectorXf::Zero(no);
		dL_dW = MatrixXf::Zero(no, ni);
	};

	// ReLU activation function for entire layer
	void activation(const VectorXf& input){
		z = (weights * input + bias).cwiseMax(0);
	};

	VectorXf backpropagate(const VectorXf& delta, const VectorXf& input){
		VectorXf mask = (z.array() > 0).cast<float>();
		VectorXf delta_relu = delta.cwiseProduct(mask);

		MatrixXf dL_dW = delta_relu * input.transpose();
		VectorXf dL_db = delta_relu;

		weights -= learning_rate * dL_dW;
		bias    -= learning_rate * dL_db;

		return weights.transpose() * delta;
	};
};

//Linear Layer class - i'll be using this as the output layer
class Linear_Layer{
public:
	MatrixXf weights;
	VectorXf bias;
	VectorXf z;
	MatrixXf dL_dW;
	int ni; // Number of neurons inside previous layer
	int no; // Number of neurons inside layer

	Linear_Layer(int ni, int no) : ni(ni), no(no){
		weights = MatrixXf::Random(no, ni);
		bias = VectorXf::Zero(no);
		z = VectorXf::Zero(no);
		dL_dW = MatrixXf::Zero(no, ni);
	};

	// Linear activation funcion for entire layer
	void activation(const VectorXf& input){
		z = weights * input + bias;
	};

	VectorXf backpropagate(const VectorXf& y, const VectorXf& input){
		dL_dW = (z - y) * input.transpose();
		VectorXf dL_db = (z - y);

		weights -= learning_rate * dL_dW;
		bias -= learning_rate * dL_db;

		return weights.transpose() * dL_db;
	}
};

int main() {
    ReLU_Layer hidden(2, 4);
    Linear_Layer output(4, 1);

    std::vector<VectorXf> X;
    std::vector<VectorXf> Y;

    for(int i = 0; i < 100; i++){
        VectorXf x(2);
        x << float(rand() % 10), float(rand() % 10);
        VectorXf y(1);
        y << 2*x(0) + 3*x(1);
        X.push_back(x);
        Y.push_back(y);
    }

    for(int epoch = 0; epoch < 10000; epoch++){
        float loss = 0;
        for(int i = 0; i < X.size(); i++){
            hidden.activation(X[i]);
            output.activation(hidden.z);

            loss += (output.z - Y[i]).squaredNorm();

            VectorXf delta = output.backpropagate(Y[i], hidden.z);
            hidden.backpropagate(delta, X[i]);
        }
        if(epoch % 100 == 0)
            std::cout << "Epoch " << epoch << " loss: " << loss << std::endl;
    }

    VectorXf test(2);
    test << 3, 4;
    hidden.activation(test);
    output.activation(hidden.z);
    std::cout << "Prediction for (3,4): " << output.z(0) << " | Expected: " << 2*3 + 3*4 << std::endl;

    return 0;
}