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

const int FEATURE_COUNT = 201;
//const int TRAINING_SET_SIZE = 1866819;

/*test*/const int TRAINING_SET_SIZE = 100;

const int TESTING_SET_SIZE = 282796;
const int SELECTED_COUNT = 12;//   select one feature from random 12 features
const int MAX_NODE_COUNT = 4096;//  tree size
const double OK_RATIO = 0.8;

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

	Node():feature_index(-1), left(-1), right(-1), label(-1) {}
};

struct Triple {
	double value;
	int feature_index;
	double gini;

	Triple(): value(0.0), feature_index(0), gini(105.0) {}
	Triple(double v, int i, double g): value(v), feature_index(i), gini(g) {}
};

vector<Record> data_set(TRAINING_SET_SIZE);

clock_t start() {
	return clock();
}

int stop(clock_t start) {
	clock_t finish = clock();
	return (double) (finish - start) / CLOCKS_PER_SEC;
}

void read(string path, vector<Record> &set) {
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

		///// debug
//		  cout << it->label << " ";
//		  for (int i = 0; i < FEATURE_COUNT; ++i) if (it->features[i] != 0) cout << i + 1 << ":" << it->features[i] << " ";
//		  cout << endl;
		///////////////

		it++;
	}
	file.close();
}

vector<int> getRandomFeatures(int count) {
	vector<int> v(count);
	for (int i = 0; i < count; ++i) {
		v[i] = rand() % FEATURE_COUNT;
	}
	return v;
}

Triple findBestFeature(const list<int> &data_indices, const vector<int> &features) {
	vector<Triple> p;// store the best threshold of each feature
	set<double> values;//  store possible thresholds of each feature
	list<int>::const_iterator it;
	set<double>::iterator it_set;

	//  different features
	assert(features.size() == SELECTED_COUNT);

	for (int i = 0; i < features.size(); ++i) {
		
		int selected = features[i];// feature
		
		//  record every possible threshold
		for (it = data_indices.begin(); it != data_indices.end(); ++it) {
			values.insert(data_set[*it].features[selected]);
		}
		
		double gini_min = 100.0;
		double best_threshold = 0.0;
		
		//  different thresholds
		for (it_set = values.begin(); it_set != values.end(); ++it_set) {
			int less_false = 0;
			int less_true = 0;
			int greater_false = 0;
			int greater_true = 0;

			// counting
			for (it = data_indices.begin(); it != data_indices.end(); ++it) {
				if (data_set[*it].features[selected] < *it_set)
					data_set[*it].label == 0 ? less_false++ : less_true++;
				else
					data_set[*it].label == 0 ? greater_false++ : greater_true++;
			}
			cout << "threshold " << *it_set <<  " counting result: " << data_indices.size() << endl
				<< less_false << endl
				<< less_true << endl
				<< greater_false << endl
				<< greater_true << endl;


			//  calculate gini index;
			double t = less_false + less_true;
			double t2 = data_indices.size() - t;

			double m1 = 0.0;
			double m2 = 0.0;
			
			if (t != 0.0) {
				m1 = less_false * 1.0 / t;
				m2 = less_true * 1.0 / t;
			}
			double l = 1.0 - m1 * m1 - m2 * m2;

//			cout << "m1, m2 " << m1 << ", " << m2 << endl;


			m1 = m2 = 0.0;

			if (t2 != 0.0) {
				m1 = greater_false * 1.0 / t2;
				m2 = greater_true * 1.0 / t2;
			}
			double r = 1.0 - m1 * m1 - m2 * m2;

//			cout << "m1, m2 " << m1 << ", " << m2 << endl << endl;
			
			if (r + l < gini_min) {
				gini_min = r + l;
				best_threshold = *it_set;
			}


			// if (r < gini_min) {
			// 	gini_min = r;
			// 	best_threshold = *it_set;
			// }

			// if (l < gini_min) {
			// 	gini_min = l;
			// 	best_threshold = *it_set;
			// }

		}

		p.push_back(Triple(best_threshold, selected, gini_min));
		values.clear();
		
		//debug
		cout << "feature " << selected <<"\'s smallest gini index: " << gini_min << " threshold: "<< best_threshold << endl << endl;;
	}

	Triple pp;
	for (int i = 0; i < p.size(); ++i) {
		if (pp.gini > p[i].gini) pp = p[i];
	}

	cout << "best feature: " << pp.feature_index << ", threshold: " << pp.value << endl;

	return pp;
}

list<int> spilt(vector<Node> &dt, int root, int &next, list<int> &data_indices, const Triple &p) {
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
	for (list<int>::iterator it = data_indices.begin(); it != data_indices.end();) {
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

	// judge
	double left_ratio = 0.0;
	if (less_false + less_true != 0) left_ratio = less_false * 1.0 / (less_false + less_true);// the proportion of label 0
	double right_ratio = 0.0;
	if (greater_false + greater_true != 0) greater_false * 1.0 / (greater_false + greater_true);// the proportion of label 0
	
	dt[root].left = next;
	if (left_ratio < 1 - OK_RATIO) dt[next].label = 1;
	else if (left_ratio > OK_RATIO) dt[next].label = 0;
	next++;

	dt[root].right = next;
	if (right_ratio < 1 - OK_RATIO) dt[next].label = 1;
	else if (right_ratio > OK_RATIO) dt[next].label = 0;
	next++;
	
	cout << "left: " << left_ratio << " right: " << right_ratio << endl << endl;

	return right_side;
}

void build(vector<Node> &dt, int root, int &next, list<int> data_indices) {
	// !!
//	cout << "build: " << root << " " << next << " " << dt[root].label << endl;
	
	if (next < MAX_NODE_COUNT && root != -1 && dt[root].label == -1) {
		vector<int> features = getRandomFeatures(SELECTED_COUNT);

		Triple p = findBestFeature(data_indices, features);

		cout << p.feature_index << " " << p.gini << " " << p.value << endl;

		list<int> right_side = spilt(dt, root, next, data_indices, p);//  spilt the data and add new nodes

		
		build(dt, dt[root].left, next, data_indices);
		build(dt, dt[root].right, next, right_side);
	}
}

void buildUp(vector<Node> &dt) {
	list<int> indices;
	int next = 1;
	//  construct data indices
	for (int i = 0; i < TRAINING_SET_SIZE; ++i) indices.push_back(i);
	
	build(dt, 0, next, indices);
}

void print(vector<Node> &dt) {
	fstream file("dt.txt", fstream::out);
	file << "feature_index value label left right\n";
	for (int i = 0; i < dt.size(); ++i) {
		file << dt[i].feature_index << " "
			<< dt[i].value << " "
			<< dt[i].label << " "
			<< dt[i].left << " "
			<< dt[i].right << endl;
	}
}

void classify(vector<Node> &dt) {
	fstream result("result.txt", fstream::out);

	for (int i = 0; i < data_set.size(); ++i) {
		int ptr = 0;
		while (dt[ptr].label == -1) {
			Node node = dt[ptr];
			if (data_set[i].features[node.feature_index] < node.value) ptr = node.left;
			else ptr = node.right;
		}
		result << i << "," << dt[ptr].label << endl;
	}
	result.close();
}

int main() {
	srand(time(NULL));
	clock_t t;
	vector<Node> dt(MAX_NODE_COUNT);
	
	t = start();
	cout << "reading training set...\t";
    read("train_data.txt", data_set);
    cout << stop(t) << "s" << endl;

	t = start();
	cout << "building up...\t";
    buildUp(dt);
    cout << stop(t) << "s" << endl;

    print(dt);


    data_set.resize(TESTING_SET_SIZE);
//    read("test_data.txt", data_set);

    // classify(dt);

//	system("pause");
    return 0;
}
