#include <iostream>
#include <vector>
#include <list>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sstream>

using namespace std;

const int dim = 201;
const int training_size = 1866819;
const int testing_size = 282796;
const int feature_count = 20;

struct Record {
	vector<double> features;
	double label;

	Record():features(vector<double>(dim, 0)), label(0) {}
};

struct Node {
	int index;
	double value;// test value < value
	int left;// 0
	int right;// 1

	Node(): left(-1), right(-1) {}
};

list<Record> training_set(training_size);
list<Record> used;
list<Record> testing_set(testing_size);

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
		// for (int i = 0; i < dim; ++i) cout << i + 1 << ":" << it->features[i] << " ";
		// cout << endl;
		/////////////////

		it++;
	}
	file.close();
}

// inline vector<int> randomFeatures(int count) {
// 	vector<int> feature_indices(count);
// 	for (int i = 0; i < count; ++i) feature_index[i] = rand() % dim;
// 	return feature_indices;
// }

// double spilt(int index, bool &flag);

// vector<Node> training(int count) {
// 	vector<int> indices = randomFeatures(count);
// 	vector<Node> tree(count);
// 	bool flag;
// 	for (int i = 0; i < count; ++i) {
// 		tree[i].index = randomFeatures[i];
// 		tree[i].value = spilt(randomFeatures[i], flag);
// 		if (flag) tree[i].left = i + 1;
// 		else tree[i].right = i + 1;
// 	}
// 	tree[i].left = tree[i].right = -1;
// }

int main() {
	srand(time(NULL));
	
	clock_t t = start();
    read("train_data.txt", training_size, training_set);
    cout << "read training set: " << stop(t) << "s" << endl;

    // read("test_data.txt", testing_size, testing_set);
    return 0;
}
