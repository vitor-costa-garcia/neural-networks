#include <stdio.h>

#define LR 0.1
#define EPOCHS 100

struct perceptron{
	float bias;
	float weights[2];
};

int predict(struct perceptron *p, int input[2]){
	float z = p -> bias;
	for(int i = 0 ; i < 2; i++){
		z += p -> weights[i] * input[i];
	}
	return z >= 0 ? 1 : 0;
};

void update(struct perceptron *p, int input[2], int output){
	int predicted_val = predict(p, input);
	if(predicted_val != output){
		p -> bias += LR*(output - predicted_val);
		for(int i = 0 ; i < 2; i++){
			p -> weights[i] += LR*(output - predicted_val)*input[i];
		}
	}
	return;
};

void fit(struct perceptron *p, int epochs, int inputs[4][2], int outputs[4], int n){
	for(int e = 0; e < epochs; e++){
		for(int i = 0; i < n; i++){
			update(p, inputs[i], outputs[i]);
		}
	}
};

int main(){
	struct perceptron p = {
		.bias = 0.0f,
		.weights = {0.0f, 0.0f}
	};

	int inputs[4][2] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	int outputs[4] = {
		0,
		1,
		1,
		1
	};

   	printf("Perceptron Weights: [%f] [%f]\n{0, 0}: %d\n{0, 1}: %d\n{1, 0}: %d\n{1, 1}: %d\n",
		p.weights[0], p.weights[1],
		predict(&p, inputs[0]),
		predict(&p, inputs[1]),
		predict(&p, inputs[2]),
		predict(&p, inputs[3])
	);


	fit(&p, EPOCHS, inputs, outputs, 4);

	printf("Perceptron Weights: [%f] [%f]\n{0, 0}: %d\n{0, 1}: %d\n{1, 0}: %d\n{1, 1}: %d\n",
		p.weights[0], p.weights[1],
		predict(&p, inputs[0]),
		predict(&p, inputs[1]),
		predict(&p, inputs[2]),
		predict(&p, inputs[3])
	);
};



