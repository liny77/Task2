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
//const int TRAINING_SET_SIZE = 1866819;

const int TRAINING_SET_SIZE = 5000;

const int TESTING_SET_SIZE = 282796;
const int SELECTED_COUNT = 12;//   select one feature from random 12 features
const int MAX_NODE_COUNT = 4096;//  tree size
const int OK_RATIO = 0.8;

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

	Node(): left(-1), right(-1), label(-1) {}
};

struct Pair {
	double value;
	int feature_index;

	Pair(): value(0), feature_index(0) {}
	Pair(double v, int i): value(v), feature_index(i) {}
};

vector<Record> data_set(TRAINING_SET_SIZE);

clock_t start() {
	return clock();
}

int stop(clock_t start) {
	clock_t finish = clock();
	return (double) (finish - start) / CLOCKS_PER_SEC;
}

void read(string path, int size, vector<Record> &set) {
	ifstream file(path.c_str());
	string line;
	stringstream ss;
	int index;
	char colon;
	vector<Record>::iterator it = set.begin();
	while (it != set.end() && !getline(file, line).eof()) {
		ss.str(line);
		ss.clear();
		ss >> it->label;
		while (ss >> index) {
			ss >> colon >> it->features[index - 1];
		}

		///////// debug
		// cout << it->label << " ";
		// for (int i = 0; i < 10; ++i) cout << i + 1 << ":" << it->features[i] << " ";
		// cout << endl;
		/////////////////

		it++;
	}
	file.close();
}

vector<int> getRandomFeatures(int count) {
	vector<int> v(count);
	for (int i = 0; i < count; ++i) {
		v[i] = rand() % count;
	}
	return v;
}

Pair findBestFeature(list<int> &set, const vector<int> &features) {
	vector<Pair> p;// store the best threshold of each feature
	vector<double> values;//  store possible thresholds of each feature
	vector<double> gini;//  store the gini index of each threshold of each feature
	list<int>::iterator it;

	//  different features
	for (int i = 0; i < features.size(); ++i) {
		values.clear();
		
		//  record every possible threshold
		for (it = set.begin(); it != set.end(); ++it) {
			if (data_set[*it].features[i] != 0) values.push_back(data_set[*it].features[i]);
		}

		gini.clear();
		//  different thresholds
		for (int a = 0; a < values.size(); ++a) {
			int less_false = 0;
			int less_true = 0;
			int greater_false = 0;
			int greater_true = 0;

			// counting
			for (it = set.begin(); it != set.end(); ++it) {
				if (data_set[*it].features[features[i]] < values[a])
					data_set[*it].label == 0 ? less_false++ : less_true++;
				else
					data_set[*it].label == 0 ? greater_false++ : greater_true++;
			}

			//  cacluate gini index and store;
			int t = less_false + less_true;
			int t2 = set.size() - t;
			double l = 1.0 - (double) (less_false * less_false + less_true * less_true)
											/ (double) (t * t);
			double r = 1.0 - (double) (greater_false * greater_false + greater_true * greater_true)
											/ (double) (t2 * t2);
			gini.push_back(r + l);
		}

		double min = gini[0];
		for (int a = 0; a < gini.size(); a++) {
			if (min > gini[a]) min = gini[a];
		}
		p.push_back(Pair(min, features[i]));
	}

	Pair pp = p[0];
	for (int i = 0; i < p.size(); ++i) {
		if (pp.value > p[i].value) pp = p[i];
	}

	return pp;
}

list<int> spilt(vector<Node> &dt, int root, int &next, list<int> &all, const Pair &p) {
	dt[root].value = p.value;
	dt[root].feature_index = p.feature_index;

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
	for (list<int>::iterator it = all.begin(); it != all.end();) {
		if (data_set[*it].features[p.feature_index] < p.value) {
			data_set[*it].label == 0 ? less_false++ : less_true++;
			++it;
		}
		else {
			data_set[*it].label == 0 ? greater_false++ : greater_true++;
			right_side.push_back(*it);
			it = all.erase(it);
		}
	}

	// judge
	double left_ratio = less_false * 1.0 / (less_false + less_true);// the proportion of label 0
	double right_ratio = greater_false * 1.0 / (greater_false + greater_true);// the proportion of label 0
	
	dt[root].left = next;
	if (left_ratio < 1 - OK_RATIO) dt[next++].label = 1;
	else if (left_ratio > OK_RATIO) dt[next++].label = 0;

	dt[root].right = next;
	if (right_ratio < 1 - OK_RATIO) dt[next++].label = 1;
	else if (right_ratio > OK_RATIO) dt[next++].label = 0;

	return right_side;
}

void build(vector<Node> &dt, int root, int &next, list<int> set) {
	// !!
	if (next < MAX_NODE_COUNT && root != -1 && dt[root].label == -1) {
		vector<int> features = getRandomFeatures(FEATURE_COUNT);
		Pair p = findBestFeature(set, features);

		list<int> right_side = spilt(dt, root, next, set, p);//  spilt the data and add new nodes

		build(dt, dt[root].left, next, set);
		build(dt, dt[root].right, next, right_side);
	}
}

void buildUp(vector<Node> &dt) {
	list<int> indices;
	int next = 1;
	for (int i = 0; i < TRAINING_SET_SIZE; ++i) indices.push_back(i);
	build(dt, 0, next, indices);
}

void print(vector<Node> &dt, int i) {
	if (dt[i].label == -1) {
		cout << dt[i].feature_index << ":" << dt[i].value << endl;
		cout << "left ";
		print(dt, dt[i].left);
		cout << "right ";
		print(dt, dt[i].left);
	}
}

int main() {
	srand(time(NULL));
	clock_t t;
	vector<Node> dt(MAX_NODE_COUNT);

	t = start();
	cout << "reading training set...\t";
    read("train_data.txt", TRAINING_SET_SIZE, data_set);
    cout << stop(t) << "s" << endl;

	t = start();
	cout << "build up...\t";
    buildUp(dt);
    cout << stop(t) << "s" << endl;

    print(dt, 0);

    // read("test_data.txt", TESTING_SET_SIZE, data_set);


    return 0;
}
