#include <iostream>
#include <vector>
#include <list>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sstream>

using namespace std;

const int FEATURE_COUNT = 201;
const int TRAINING_SET_SIZE = 1866819;
const int TESTING_SET_SIZE = 282796;
const int SELECTED_COUNT = 12;
const int MAX_NODE_COUNT = 4096;  //  2^12

struct Record {
	vector<double> features;
	double label;

	Record():features(vector<double>(FEATURE_COUNT, 0)), label(0) {}
};



struct Node {
	int index;
	double value;// test value < value
	int label;
	int left;// 0
	int right;// 1
	bool good;

	Node(): left(-1), right(-1), label(-1), good(false) {}
};

list<Record> data_set(TRAINING_SET_SIZE);

clock_t start() {
	return clock();
}

int stop(clock_t start) {
	clock_t finish = clock();
	return (finish - start) / CLOCKS_PER_SEC;
}

void read(string path, int size, list<Record> &set) {
	ifstream file(path.c_str());
	string line;
	stringstream ss;
	int index;
	char colon;
	list<Record>::iterator it = set.begin();
	while (it != set.end() && !getline(file, line).eof()) {
		ss.str(line);
		ss.clear();
		ss >> it->label;
		while (ss >> index) {
			ss >> colon >> it->features[index - 1];
		}

		///////// debug
		// cout << it->label << " ";
		// for (int i = 0; i < FEATURE_COUNT; ++i) cout << i + 1 << ":" << it->features[i] << " ";
		// cout << endl;
		/////////////////

		it++;
	}
	file.close();
}

int judge(int zero, int one) {
	// 1 --> true
	// 0 --> false
	// -1 --> unknown
	int sum = zero + one;
	if (zero * 1.0 / sum >= 0.75) return 0;
	else if (one * 1.0 / sum >= 0.75) return 1;
	else return -1;
}

double spilt(vector<Node> &tree, int node_index) {
	return 0.0;
}

vector<Node> training() {
	int label;
	int counter = 0;
	vector<Node> tree(MAX_NODE_COUNT);
	for (int i = 0; i < MAX_NODE_COUNT && counter < SELECTED_COUNT; ++i) {
		if ((i - 1) / 2 >= 0 && tree[(i - 1) / 2].good) continue;
		//  select a feature
		tree[i].index = rand() % FEATURE_COUNT;
		tree[i].value = spilt(tree, i);
		int son = 2 * i + 1;
		if (son < MAX_NODE_COUNT && son + 1 < MAX_NODE_COUNT) {
			tree[i].left = son;
			tree[i].right = son + 1;
		}
		counter++;
	}
	return tree;
}

int main() {
	srand(time(NULL));
	
	clock_t t = start();
    read("train_data.txt", TRAINING_SET_SIZE, data_set);
    cout << "read training set: " << stop(t) << "s" << endl;

    // read("test_data.txt", TESTING_SET_SIZE, data_set);
    return 0;
}
