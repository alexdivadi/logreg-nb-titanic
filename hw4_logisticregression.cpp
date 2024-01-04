/*
title: "Homework 4: Logistic Regression"
subtitle: "4375 Machine Learning with Dr. Mazidi"
author: "David Allen"
date: "February 24, 2022"
purpose: to recreate logistic regression algorithm from scratch in C++
requirements: C++ env, hw4_logisticregression.cpp, titanic_projcet.csv
*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <chrono>

using namespace std;

/*
name: "Sigmoid"
params: vector<double> x
returns: vector<double> sigmoid
purpose: to get sigmoid values of vector
*/
vector<double> sigmoid(vector<double> &x) {
	vector<double> sigmoid;
	for (double &val : x) {
		sigmoid.push_back(1.0 / (1 + exp(-val)));
	}
	return sigmoid;
}

/*
name: "Predict"
params: vector<double> x
returns: vector<double> pred
purpose: to get prediction probabilities
*/
vector<double> predict(vector<double>& x) {
	vector<double> pred;
	for (double& val : x) {
		pred.push_back(exp(val) / (1 + exp(val)));
	}
	return pred;
}

/*
name: "Transpose"
params: vector<double> x, int rows, int cols
returns: vector<double> x_t
purpose: transposes single vector matrix
*/
vector<double> transpose(vector<double> &x, int rows, int cols) {
	vector<double> transposed(rows*cols, -1);
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			transposed.at(row + col*rows) = x.at(row*cols + col);
		}
	}
	return transposed;
}

/*
name: "Matrix * Column Vector"
params: vector<double> x, vector<double> y, int rows, int cols (dimensions of new matrix)
returns: vector<double> product
purpose: to multiply matrix by column vector. 
notes: could be redesigned to support matrix multiplication in general, error checking
*/
vector<double> matrix_times_col_vector(vector<double> &x, vector<double> &w, int rows, int cols) {
	vector<double> result;
	double sum;
	// outer loop should iterate per # of observations
	for (int obs = 0; obs < rows; obs++) {
		sum = 0.0;
		// inner for loop should iterate per # of weights
		for (int idx = 0; idx < cols; idx++) {
			sum += x[obs*cols+idx] * w[idx];
		}
		// push dot product to result vector
		result.push_back(sum);
	}
	return result;
}
/*
name: "Matrix * Constant"
params: vector<double> x, double c
returns: vector<double> product
purpose: to multiply matrix by a constant
*/
vector<double> matrix_times_constant(vector<double> &x, double c) {
	vector<double> result;
	for (double &val : x) {
		result.push_back(val * c);
	}
	return result;
}

/*
name: "Calc error"
params: vector<double> x, vector<double> y
returns: vector<double> difference
purpose: gets difference of each element in vectors of equal dimension
*/
vector<double> calc_error(vector<double> &labels, vector<double> &pred) {
	vector<double> error;
	for (int i = 0; i < labels.size(); i++) {
		error.push_back(labels.at(i) - pred.at(i));
	}
	return error;
}

/*
name: "Calc sum"
params: vector<double> x, vector<double> y
returns: vector<double> sum
purpose: gets sum of each element in vectors of equal dimension
*/
vector<double> calc_sum(vector<double> &x, vector<double> &y) {
	vector<double> sum;
	for (int i = 0; i < x.size(); i++) {
		sum.push_back(x.at(i) + y.at(i));
	}
	return sum;
}

/*
name: "Print vector"
params: vector<double> x
returns: none
purpose: prints each element of vector
*/
void print_vector(vector<double> &x) {
	for (double &val : x) {
		printf("%.5f ", val);
	}
	endl(cout);
}

/*
name: "Classify"
params: vector<double> x, double threshold
returns: vector<double> pred
purpose: determines classification of probs given threshold
*/
vector<double> classify(vector<double>& probs, double threshold) {
	vector<double> predictions;
	for (double &val : probs) {
		predictions.push_back(val > threshold);
	}
	return predictions;
}

/*
name: "Accuracy"
params: vector<double> y, vector<double> pred, int size
returns: double accuracy
purpose: gets accuracy of model on test predictions
*/
double accuracy(vector<double>& y, vector<double>& pred, int size) {
	double acc = 0.0;
	for (int i = 0; i < size; i++) {
		if (y[i] == pred[i])
			acc++;
	}
	return acc / size;
}
/*
name: "Sensitivity"
params: vector<double> y, vector<double> pred, int size
returns: double sensitivity
purpose: gets sensitivity of model on test predictions
*/
double sensitivity(vector<double>& y, vector<double>& pred, int size) {
	double tpr = 0.0, pr = 0.0;
	for (int i = 0; i < size; i++) {
		if (pred[i]) {
			pr++;
			if (y[i])
				tpr++;
		}
	}
	return tpr/pr;
}
/*
name: "Specificity"
params: vector<double> y, vector<double> pred, int size
returns: double specificity
purpose: gets specificity of model on test predictions
*/
double specificity(vector<double>& y, vector<double>& pred, int size) {
	double tnr = 0.0, nr = 0.0;
	for (int i = 0; i < size; i++) {
		if (!pred[i]) {
			nr++;
			if (!y[i])
				tnr++;
		}
	}
	return tnr / nr;
}


int main(int argc, char* argv[]) {
	cout << "LOGISTIC REGRESSION PREDICTION:" << endl << endl;
	ifstream inFS;
	string line;
	//Columns: "","pclass","survived","sex","age"
	string id_in, pclass_in, survived_in;
	const int MAX_LEN = 2000;
	vector<double> pclass(MAX_LEN);
	vector<double> survived(MAX_LEN);

	// Opening File:
	cout << "Opening file titanic_project.csv." << endl;

	inFS.open("titanic_project.csv");
	if (!inFS.is_open()) {
		cout << "Could not open file titanic_project.csv." << endl;
		return 1; // 1 meaning an error has occured
	}

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	// assuming first line is column headings
	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good()) {
		// get col info
		getline(inFS, id_in, ',');
		getline(inFS, pclass_in, ',');
		getline(inFS, survived_in, ',');
		getline(inFS, line, '\n');

		// assign values into appropriate vectors
		pclass.at(numObservations) = stof(pclass_in);
		survived.at(numObservations) = stof(survived_in);

		numObservations++;
	}

	pclass.resize(numObservations);
	survived.resize(numObservations);

	// *both vectors are same size
	cout << "new length " << pclass.size() << endl;

	cout << "Closing file titanic_project.csv now." << endl;
	inFS.close();

	// Review data and print summary:
	cout << "Number of records: " << numObservations << endl;

	// Perform log-reg
	// declare train-test split
	int trainSize = 900, numWeights=2;
	int testSize = numObservations - trainSize;
	vector<double> labels = vector<double>(survived.begin(), survived.begin() + trainSize);
	vector<double> y = vector<double>(survived.begin() + trainSize, survived.end());
	// declare matrices
	vector<double> weights{1, 1};
	vector<double> data_matrix(numWeights * trainSize), test_matrix(numWeights * testSize);
	// train matrix
	for (int i = 0, j = 0; i < trainSize * numWeights; i++) {
		if (i % numWeights == 0)
			data_matrix.at(i) = 1;
		else {
			data_matrix.at(i) = pclass.at(j);
			j++;
		}
	}
	// test matrix
	for (int i = 0, j = trainSize; i < testSize * numWeights; i++) {
		if (i % numWeights == 0)
			test_matrix.at(i) = 1;
		else {
			test_matrix.at(i) = pclass.at(j);
			j++;
		}
	}
	// get transpose of train matrix (instead of calculating at each iteration)
	vector<double> t_data_matrix = transpose(data_matrix, trainSize, numWeights);
	// declare vectors for later
	vector<double> error(trainSize), prob_vector(trainSize);
	vector<double> predicted(testSize), probs(testSize), pred(testSize);
	// declare training features
	double learning_rate = 0.001;
	int iterations = 5000;

	// Train Data
	printf("\nTraining data (%d iterations)...\n", iterations);

	// start timer
	chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	for (int i = 0; i < iterations; i++) {
		vector<double> temp = matrix_times_col_vector(data_matrix, weights, trainSize, numWeights);
		prob_vector = sigmoid(temp);
		error = calc_error(labels, prob_vector);
		temp = matrix_times_constant(t_data_matrix, learning_rate);
		temp = matrix_times_col_vector(temp, error, numWeights, trainSize);
		weights = calc_sum(temp, weights);
	}
	chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); // end timer

	// training results
	cout << "Training time: " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds" << endl;
	cout << "\nWeights: ";
	print_vector(weights);

	// Perform Testing
	cout << "Testing data..." << endl;
	predicted = matrix_times_col_vector(test_matrix, weights, testSize, numWeights);
	probs = predict(predicted);
	pred = classify(probs, 0.5);

	// performance results
	cout << "\nResults" << endl;
	cout << "Accuracy: " << accuracy(y, pred, testSize) << endl;
	cout << "Sensitivity: " << sensitivity(y, pred, testSize) << endl;
	cout << "Specificity: " << specificity(y, pred, testSize) << endl;


	cout << "\nProgram finished running.";
	return 0; // code terminated without errors
}