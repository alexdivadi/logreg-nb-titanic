/*
title: "Homework 4: Naive Bayes"
subtitle: "4375 Machine Learning with Dr. Mazidi"
author: "David Allen"
date: "February 25, 2022"
purpose: to recreate naive bayes algorithm from scratch in C++
requirements: C++11 env, hw4_naivebayes.cpp, titanic_projcet.csv
*/

// for PI value
#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace std;


/*
name: "Print vector"
params: vector<T> x
returns: none
purpose: prints each element of generic vector
*/
template<typename T>
void print_vector(vector<T>& x) {
	for (auto& val : x) {
		cout << val << " ";
	}
	endl(cout);
}
/*
name: "Matrix Conditional"
params: vector<T> x, vector<int> n, int sentinel_value
returns: vector<T> result
purpose: gets parallel vector of any type by checking values of another vector against sentinel
*/ 
template<typename T>
vector<T> matrix_conditional(vector<T>& x, vector<int>& n, int sentinel) {
	// return if sizes don't match
	if (x.size() != n.size()) {
		cout << "Error: vector sizes don't match" << endl;
		return vector<T>();
	}

	vector<T> result;
	for (int i = 0; i < x.size(); i++) {
		if (n[i] == sentinel)
			result.push_back(x[i]);
	}
	return result;
}
/*
name: "Sum"
params: vector<double> x
returns: double sum
purpose: to sum double values in vector
*/
double sum(vector<double>& x) {
	double sum = 0.0;
	for (double &val : x) {
		sum += val;
	}
	return sum;
}
/*
name: "Mean"
params: vector<double> x
returns: double mean
purpose: to find mean of double values in vector
*/
double mean(vector<double>& x) {
	return sum(x) / (double)x.size();
}
/*
name: "Covariance"
params: vector<double> x, vector<double> y
returns: double covar
purpose: to calculate covariance of two vectors
*/
double covar(vector<double>& x, vector<double>& y) {
	// check if vectors are same size
	if (x.size() != y.size()) {
		cout << "Error: vectors are not same size. -1 returned.";
		return -1.0;
	}

	// get averages and size
	double covar = 0.0;
	double mean_x = mean(x);
	double mean_y = mean(y);
	int size = x.size();

	// Sum product of residuals (from formula on p. 74)
	for (int i = 0; i < size; i++)
		covar += (x[i] - mean_x) * (y[i] - mean_y);

	// return sum / n-1
	return covar / (size - 1.0);
}
/*
name: "Calculate Quantitative likelihood"
params: double x, double mean_x, double var_x
returns: double lh
purpose: to calculate likelihood of quantitative value
*/
double calc_quant_lh(double x, double mean_x, double var_x) {
	// return formula
	return 1 / sqrt(2 * M_PI * var_x) * exp(-(pow(x-mean_x, 2))/(2*var_x));
}
/*
name: "Calculate Qualitative likelihood"
params: vector<int> x, vector<int> n, int i, int c
returns: double lh
purpose: to calculate likelihood of qualitative value
*/
double calc_qual_lh(vector<int> x, vector<int> n, int i, int c) {
	// return if sizes don't match
	if (x.size() != n.size()) {
		cout << "Error: vectors are not same size. -1 returned.";
		return -1.0;
	}
	// get sum of X_ic and N_c
	vector<int> N = matrix_conditional(n, n, c);
	double sum = 0;
	for (int idx = 0; idx < x.size(); idx++) {
		sum += (double)(x[idx] == i) * (double)(n[idx] == c);
	}
	return sum / N.size();
}

/*
name: "Calculate Raw Probabilities"
params: vector<double>& apriori, vector<double>& lh_pclass, vector<double>& lh_sex,
		vector<double>& age_mean, vector<double>& age_var (model data),
		int pclass, int sex, int age (predictor values)
returns: vector<double> probs
purpose: to get prediction probabilities given naive bayes model
*/
vector<double> calc_raw_prob(vector<double>& apriori, vector<double>& lh_pclass, vector<double>& lh_sex,
							 vector<double>& age_mean, vector<double>& age_var,
							 int pclass, int sex, int age) {
	vector<double> probs;
	double num_s = lh_pclass[3 + pclass - 1] * lh_sex[2 + sex] * calc_quant_lh(age, age_mean[true], age_var[true]) * apriori[true];
	double num_p = lh_pclass[pclass - 1] * lh_sex[sex] * calc_quant_lh(age, age_mean[false], age_var[false]) * apriori[false];
	double denominator = num_s + num_p;
	probs.push_back(num_p / denominator);
	probs.push_back(num_s / denominator);
	return probs;
}

/*
name: "Print matrix"
params: vector<double> x, int width
returns: none
purpose: prints each element of matrix, adding newline for end of row
*/
void print_matrix(vector<double>& x, int width) {
	for (int i = 0; i < x.size(); i++) {
		if (i % width == 0)
			endl(cout);
		printf("%.5f ", x[i]);
	}
	endl(cout);
}
/*
name: "Classify Binary Matrix"
params: vector<double> probs
returns: vector<double> pred
purpose: determines binary classification of probs by choosing maximum
*/
vector<double> classify_binary_matrix(vector<double>& probs) {
	vector<double> predictions;
	for (int i = 0; i < probs.size() / 2; i++) {
		predictions.push_back(probs[2 * i] < probs[2 * i + 1]);
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
	cout << "NAIVE BAYES PREDICTION:" << endl << endl;
	ifstream inFS;
	string line;
	//Columns: "","pclass","survived","sex","age"
	string id_in, pclass_in, survived_in, sex_in, age_in;
	const int MAX_LEN = 2000;
	vector<int> pclass(MAX_LEN), survived(MAX_LEN), sex(MAX_LEN);
	vector<double> age(MAX_LEN);

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
		// read col values
		getline(inFS, id_in, ',');
		getline(inFS, pclass_in, ',');
		getline(inFS, survived_in, ',');
		getline(inFS, sex_in, ',');
		getline(inFS, age_in, '\n');

		// assign values into appropriate vectors
		pclass.at(numObservations) = stof(pclass_in);
		survived.at(numObservations) = stof(survived_in);
		sex.at(numObservations) = stof(sex_in);
		age.at(numObservations) = stof(age_in);

		numObservations++;
	}

	pclass.resize(numObservations); sex.resize(numObservations);
	survived.resize(numObservations); age.resize(numObservations);

	// *both vectors are same size
	cout << "new length " << pclass.size() << endl;

	cout << "Closing file titanic_project.csv now." << endl;
	inFS.close();

	// Review data and print summary:
	cout << "Number of records: " << numObservations << endl;

	// Perform naive-bayes
	// declare train-test split
	int trainSize = 900, numWeights=2;
	int testSize = numObservations - trainSize;
	// train/test sets
	vector<int> survived_train(survived.begin(), survived.begin() + trainSize),
		pclass_train(pclass.begin(), pclass.begin() + trainSize),
		sex_train(sex.begin(), sex.begin() + trainSize);
	vector<double> age_train(age.begin(), age.begin() + trainSize);
	vector<double> y(survived.begin() + trainSize, survived.end());
	// declare coeff matrices
	vector<int> pclasses{ 1, 2, 3 }, sex_classes{ 0, 1 }, survived_classes{ 0, 1 };
	vector<double> lh_pclass(pclasses.size() * survived_classes.size()), lh_sex(sex_classes.size() * survived_classes.size());
	vector<double> apriori, age_mean, age_var;
	
	// Perform Training
	cout << "\nTraining data..." << endl;

	// start timer
	chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	for (int& c : survived_classes) {
		vector<double> temp = matrix_conditional(age_train, survived_train, c);
		// aprioris
		apriori.push_back((double)temp.size() / trainSize);
		// pclass and sex lh
		for (int& i : pclasses) {
			lh_pclass.at(c * pclasses.size() + i-1) = calc_qual_lh(pclass_train, survived_train, i, c);
		}
		for (int& i : sex_classes) {
			lh_sex.at(c * sex_classes.size() + i) = calc_qual_lh(sex_train, survived_train, i, c);
		}
		// age likelihoods
		age_mean.push_back(mean(temp));
		age_var.push_back(covar(temp, temp));
	}
	chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); // end timer

	// training results
	cout << "Training time: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << " microseconds" << endl;

	// print coeffs
	cout << "\nApriori: ";
	print_vector(apriori);
	cout << "Likelihoods for Passenger Class: ";
	print_matrix(lh_pclass, 3);
	cout << "Likelihoods for Sex: ";
	print_matrix(lh_sex, 2);
	cout << "Age Mean: ";
	print_vector(age_mean);
	cout << "Age Variance: ";
	print_vector(age_var);

	// Perform Testing
	cout << "\nTesting data..." << endl;
	vector<double> raw, probs, pred(testSize);
	for (int i = 0; i < testSize; i++) {
		raw = calc_raw_prob(apriori, lh_pclass, lh_sex, age_mean, age_var, 
			pclass.at(trainSize + i), sex.at(trainSize + i), age.at(trainSize + i));
		probs.insert(probs.end(), raw.begin(), raw.end());
	}
	pred = classify_binary_matrix(probs);

	// Print out some raw probs
	cout << "\nPredictions (first five):" << endl;
	for (int i = 0; i < 5; i++) {
		cout << probs[false + i*2] << " " << probs[true + i*2] << endl;
	}

	// performance results
	cout << "\nResults" << endl;
	cout << "Accuracy: " << accuracy(y, pred, testSize) << endl;
	cout << "Sensitivity: " << sensitivity(y, pred, testSize) << endl;
	cout << "Specificity: " << specificity(y, pred, testSize) << endl;


	cout << "\nProgram finished running.";
	return 0; // code terminated without errors
}