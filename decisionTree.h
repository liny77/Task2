#ifndef DECISION_TREE
#define DECISION_TREE

#include <iostream>
#include <vector>
#include <list>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <sstream>
#include <set>

using namespace std;

struct DecisionTree {
	static const int FEATURE_COUNT = 201;
	// static const int TRAINING_SET_SIZE = 1866819;

	/*test*/
	static const int TRAINING_SET_SIZE = 1000;

	// static const int TESTING_SET_SIZE = 282796;

	/*test*/
	static const int TESTING_SET_SIZE = 5000;

	static const int SELECTED_COUNT = 12;//   select 12 random features
	static const int MAX_NODE_COUNT = 4096;// max tree size
	static const double OK_RATIO = 0.9;

	struct Record {
		vector<double> features;
		double label;

		Record():features(vector<double>(FEATURE_COUNT, 0)), label(0) {}
	};

	struct Node {
		int feature_index;
		double value;//  threshold
		int label;

		int left;//  false 
		int right;//  true

		Node():feature_index(-1), value(0.0), label(-1), left(-1), right(-1) {}
	};

	struct Triple {
		double value;
		int feature_index;
		double entropy;

		Triple(): value(0.0), feature_index(0), entropy(10.0) {}
		Triple(double v, int i, double g): value(v), feature_index(i), entropy(g) {}
	};

	vector<Record> data_set;
	vector<Node> tree;
	int next;

	DecisionTree() {}

	void read(string path) {
		ifstream file(path.c_str());
		string line;
		stringstream ss;
		int index;
		char colon;
		vector<Record>::iterator it = data_set.begin();
		while (it != data_set.end() && !getline(file, line).eof()) {
			ss.str(line);
			ss.clear();
			ss >> it->label;
			while (ss >> index) {
				ss >> colon >> it->features[index - 1];
			}

			///// debug
			// cout << it->label << " ";
			// for (int i = 0; i < FEATURE_COUNT; ++i) if (it->features[i] != 0)
			// 	cout << i + 1 << ":" << it->features[i] << " ";
			// cout << endl;
			///////////////

			it++;
		}
		file.close();
	}

	void readTrainData(string path) {
		data_set.resize(TRAINING_SET_SIZE);
		read(path);
	}

	void readTestData(string path) {
		data_set.resize(TESTING_SET_SIZE);
		read(path);
	}

	double calculateEntropy(int n, int p) {
		double total = n + p;
		double npro = n / total;
		double ppro = p / total;

		return -(npro * log(npro) + ppro * log(ppro));
	}

	double calculate(int n1, int p1, int n2, int p2) {
		double total = n1 + p1 + n2 + p2;
		double ig1 = (n1 && p1) ? calculateEntropy(n1, p1) : 0;
		double ig2 = (n2 && p2) ? calculateEntropy(n2, p2) : 0;

		return ig1 * ((n1 + p1) / total) + ig2 * ((n2 + p2) / total);
	}
	
	vector<int> getRandomFeatures() {
		vector<int> v(SELECTED_COUNT);
		for (int i = 0; i < v.size(); ++i) {
			v[i] = rand() % FEATURE_COUNT;
		}
		return v;
	}

	void training() {
		list<int> indices;
		tree.resize(MAX_NODE_COUNT);
		next = 1;
		// construct data indices
		for (int i = 0; i < TRAINING_SET_SIZE; ++i)
			indices.push_back(i);
		build(0, indices);
	}


	void build(int root, list<int> data_indices) {
		if (next < MAX_NODE_COUNT && root != -1 && tree[root].label == -1) {
			vector<int> rand_features = getRandomFeatures();
			Triple result = findBestFeature(data_indices, rand_features);
			
			 // cout << result.feature_index << " " << result.entropy << " " << result.value << endl;
			
			list<int> right_side = spilt(root, data_indices, result);//  spilt the data and add new nodes

			build(tree[root].left, data_indices);
			build(tree[root].right, right_side);
		}
	}

	Triple findBestFeature(const list<int> &data_indices, const vector<int> &rand_features) {
		vector<Triple> p;// store the best result of each feature
		set<double> values;// store possible thresholds of each feature

		list<int>::const_iterator it_indices;
		set<double>::iterator it_values;

		assert(rand_features.size() == SELECTED_COUNT);

		// different features
		for (int i = 0; i < rand_features.size(); ++i) {
			int selected = rand_features[i];
			// record every possible threshold
			for (it_indices = data_indices.begin(); it_indices != data_indices.end(); ++it_indices) {
				values.insert(data_set[*it_indices].features[selected]);
			}

			double entropy_MIN = 10.0;
			double best_threshold = 0.0;

			//  different thresholds
			for (it_values = values.begin(); it_values != values.end(); ++it_values) {
				int less_false = 0;
				int less_true = 0;
				int greater_false = 0;
				int greater_true = 0;

				// counting
				for (it_indices = data_indices.begin(); it_indices != data_indices.end(); ++it_indices) {
					if (data_set[*it_indices].features[selected] < *it_values)
						data_set[*it_indices].label == 0 ? less_false++ : less_true++;
					else
						data_set[*it_indices].label == 0 ? greater_false++ : greater_true++;
				}

				// cout << "threshold " << *it_values
				// 	<< " counting result: "
				// 	<< data_indices.size() << endl
				// 	<< less_false << endl
				// 	<< less_true << endl
				// 	<< greater_false << endl
				// 	<< greater_true << endl;

				double entropy = calculate(less_false, less_true, greater_false, greater_true);
				
				// cout << entropy << endl;

				if (entropy < entropy_MIN) {
					entropy_MIN = entropy;
					best_threshold = *it_values;
				}
			}

			p.push_back(Triple(best_threshold, selected, entropy_MIN));
			values.clear();
			
			// debug
			// cout << "feature " << selected <<"\'s smallest gini index: " << entropy_MIN << " threshold: "<< best_threshold << endl << endl;;
		}

		Triple pp;
		for (int i = 0; i < p.size(); ++i) {
			if (p[i].entropy < pp.entropy) pp = p[i];
		}

		 // cout << "best feature: " << pp.feature_index << ", threshold: " << pp.value << endl;

		return pp;
	}

	list<int> spilt(const int& root, list<int>& data_indices, const Triple& p) {
		tree[root].value = p.value;
		tree[root].feature_index = p.feature_index;

		list<int> right_side;
		int less_false = 0;
		int less_true = 0;
		int greater_false = 0;
		int greater_true = 0;
		/*
							< value
							/   \
						   /     \
					less_false    greater_false
					less_true	  greater_true
		*/
		for (list<int>::iterator it = data_indices.begin(); it != data_indices.end(); ) {
			if (data_set[*it].features[p.feature_index] < p.value) {
				data_set[*it].label == 0 ? less_false++ : less_true++;
				++it;
			}
			else {
				data_set[*it].label == 0 ? greater_false++ : greater_true++;
				right_side.push_back(*it);
				it = data_indices.erase(it);
			}
		}
		
		// cout << "l " << less_true + less_false << ", r " << greater_true + greater_false << endl
		//  	<< less_false << endl
		//  	<< less_true << endl
		//  	<< greater_false << endl
		//  	<< greater_true << endl;

		// calculate ratios
		double left_ratio = 0.0;
		if (less_false || less_true)
			left_ratio = less_false * 1.0 / (less_false + less_true);// the proportion of label 0
		double right_ratio = 0.0;
		if (greater_false || greater_true)
			right_ratio = greater_false * 1.0 / (greater_false + greater_true);// the proportion of label 0

		if (next < MAX_NODE_COUNT) {
			tree[root].left = next;
			if (left_ratio < 1 - OK_RATIO) tree[next].label = 1;
			else if (left_ratio > OK_RATIO) tree[next].label = 0;
			next++;
		} else {
			if (left_ratio < 0.5) tree[root].label = 1;
			else tree[root].label = 0;
		}

		if (next < MAX_NODE_COUNT) {
			tree[root].right = next;
			if (right_ratio < 1 - OK_RATIO) tree[next].label = 1;
			else if (right_ratio > OK_RATIO) tree[next].label = 0;
			next++;
		} else {
			if (right_ratio < 0.5) tree[root].label = 1;
			else tree[root].label = 0;
		}

		cout << "left: " << left_ratio << " right: " << right_ratio << endl << endl;

		return right_side;
	}

	void classify() {
		fstream result("result.txt", fstream::out);
		result << "id,label\n";

		for (int i = 0; i < data_set.size(); ++i) {
			int ptr = 0;
			while (ptr < MAX_NODE_COUNT && tree[ptr].label == -1) {
				Node node = tree[ptr];
				if (data_set[i].features[node.feature_index] < node.value) {
					if (node.left == -1) break;
					ptr = node.left;
				}
				else {
					if (node.right == -1) break;
					ptr = node.right;
				}
			}
			result << i << "," << tree[ptr].label << endl;
		}
		result.close();
	}

	void print() {
		fstream file("tree.txt", fstream::out);
		file << "feature_index value label left right\n";
		for (int i = 0; i < tree.size(); ++i) {
			file << tree[i].feature_index << " "
				<< tree[i].value << " "
				<< tree[i].label << " "
				<< tree[i].left << " "
				<< tree[i].right << endl;
		}
	}
};

#endif
