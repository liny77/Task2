#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <sstream>
#include <set>
#include <random>
#include <iomanip>
#include <thread>

using namespace std;
using namespace std::chrono;

const int FEATURE_COUNT = 201;
const int TESTING_SET_SIZE = 282796;
const int TRAINING_SET_SIZE = 1866819;
const int USED_FOR_TRAINING = 10000;
const double OK_RATIO = 1;
const int FOREST_SIZE = 128;

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

struct Vote {
    int negtive_count;
    int positive_count;

    Vote(): negtive_count(0), positive_count(0) {}

    Vote& operator+=(Vote a) {
        this->negtive_count += a.negtive_count;
        this->positive_count += a.positive_count;
        return *this;
    }
};

vector<Record> data_set;

void read(string path, int size) {
    data_set.resize(size);
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
        // /// debug
        // cout << it->label << " ";
        // for (int i = 0; i < FEATURE_COUNT; ++i) if (it->features[i] != 0)
        // cout << i + 1 << ":" << it->features[i] << " ";
        // cout << endl;
        // /////////////
        it++;
        }
    file.close();
}

random_device rd;

struct DecisionTree {
    static const int SELECTED_COUNT = 40;   
    static const int level = 10;
    static const int MAX_NODE_COUNT = 1024;// max tree size

    struct Triple {
        double value;
        int feature;
        double entropy;

        Triple(): value(0.0), feature(0), entropy(100000.0) {}
        Triple(double v, int f, double e): value(v), feature(f), entropy(e) {}
    };

    vector<Node> tree;
    int next;

    DecisionTree() {}

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
    
    set<int> getRandomFeatures() {
        set<int> features;
        while (features.size() < SELECTED_COUNT)
            features.insert(rd() % FEATURE_COUNT);
        return features;
    }

    static set<int> getDataRandomly(int count) {
        set<int> data;
        while (data.size() < count)
            data.insert(rd() % TRAINING_SET_SIZE);
        return data;
    }

    void training(int count, int i) {
        tree.resize(MAX_NODE_COUNT);
        next = 1;
        set<int> data = getDataRandomly(count);
        build(0, data, 0);
        cout << "tree " << i << "......done" << endl;
    }

    void build(int root, set<int> data, int l) {
        if (!data.empty() && root != -1 && tree[root].label == -1 && l < level) {
            set<int> rand_features = getRandomFeatures();
            Triple best_feature_info = findBestFeature(data, rand_features);
            set<int> right_side = spilt(root, data, best_feature_info, l);//  spilt the data and add new nodes
            l++;
            build(tree[root].left, data, l);
            build(tree[root].right, right_side, l);
        }
    }

    Triple findBestFeature(const set<int> &data, const set<int> &rand_features) {
        Triple best;// store the best result of each feature
        set<double> values;// store possible thresholds of each feature
        set<double>::iterator it_values;

        set<int>::const_iterator const_it_data;

        // for different features
        for (set<int>::const_iterator it = rand_features.begin(); it != rand_features.end(); ++it) {
            int selected = *it;
            // record every possible threshold
            for (const_it_data = data.begin(); const_it_data != data.end(); ++const_it_data) {
                values.insert(data_set[*const_it_data].features[selected]);
            }

            double entropy_MIN = 10000.0;
            double best_threshold = 0.0;

            //  different thresholds
            for (it_values = values.begin(); it_values != values.end(); ++it_values) {
                int less_false = 0;
                int less_true = 0;
                int greater_false = 0;
                int greater_true = 0;

                // counting
                for (const_it_data = data.begin(); const_it_data != data.end(); ++const_it_data) {
                    if (data_set[*const_it_data].features[selected] < *it_values)
                        data_set[*const_it_data].label == 0 ? less_false++ : less_true++;
                    else
                        data_set[*const_it_data].label == 0 ? greater_false++ : greater_true++;
                }

                double entropy = calculate(less_false, less_true, greater_false, greater_true);
                // cout << entropy << endl;

                if (entropy < entropy_MIN) {
                    entropy_MIN = entropy;
                    best_threshold = *it_values;
                }
            }

            if (entropy_MIN < best.entropy) {
                best.entropy = entropy_MIN;
                best.value = best_threshold;
                best.feature = selected;
            }

            values.clear();
        }
        // cout << best.entropy << " " << best.value << " " << best.feature << endl;
        return best;
    }

    set<int> spilt(const int& root, set<int>& data, const Triple& best, int l) {
        tree[root].value = best.value;
        tree[root].feature_index = best.feature;

        set<int> right_side;
        int less_false = 0;
        int less_true = 0;
        int greater_false = 0;
        int greater_true = 0;
        /*
                            < value
                            /   \
                           /     \
                    less_false    greater_false
                    less_true      greater_true
        */
        for (set<int>::iterator it = data.begin(); it != data.end(); ) {
            if (data_set[*it].features[best.feature] < best.value) {
                data_set[*it].label == 0 ? less_false++ : less_true++;
                ++it;
            }
            else {
                data_set[*it].label == 0 ? greater_false++ : greater_true++;
                right_side.insert(*it);
                it = data.erase(it);
            }
        }
        
        // cout << "l " << less_true + less_false << ", r " << greater_true + greater_false << endl
        //      << less_false << endl
        //      << less_true << endl
        //      << greater_false << endl
        //      << greater_true << endl;

        // calculate ratios
        double left_ratio = 0.0;
        if (less_false || less_true)
            left_ratio = less_false * 1.0 / (less_false + less_true);// the proportion of label 0

        double right_ratio = 0.0;
        if (greater_false || greater_true)
            right_ratio = greater_false * 1.0 / (greater_false + greater_true);// the proportion of label 0

        if (l == level - 1) {
            double total = less_false + less_true + greater_false + greater_true;
            double node_ratio = (less_false + greater_false) * 1.0 / total;
            if (node_ratio < 0.5) tree[root].label = 1;
            else if (node_ratio > 0.5) tree[root].label = 0;
            else tree[root].label = rd() % 2;
        } else {
            tree[root].left = next;
            if (left_ratio == 1) tree[next].label = 0;
            else if (left_ratio == 0) tree[next].label = 1;
            next++;

            tree[root].right = next;
            if (right_ratio == 1) tree[next].label = 0;
            else if (right_ratio == 0) tree[next].label = 1;
            next++;
        }

        // cout << "left: " << left_ratio << " right: " << right_ratio << " " << next << endl << endl;

        return right_side;
    }

    Vote voteOne(int index) {
        int ptr = 0;
        Vote v;
        while (ptr < MAX_NODE_COUNT && tree[ptr].label == -1) {
            Node node = tree[ptr];
            if (data_set[index].features[node.feature_index] < node.value) {
                if (node.left == -1) break;
                ptr = node.left;
            }
            else {
                if (node.right == -1) break;
                ptr = node.right;
            }
        }
        int label = tree[ptr].label;
        if (label == 0) v.negtive_count++;
        else v.positive_count++;
        return v;
    }

    void classify(vector<Vote> &votes) {
        for (int i = 0; i < votes.size(); ++i) {
            votes[i] += voteOne(i);
        }
    }
};

steady_clock::time_point start() {
    return steady_clock::now();
}

int stop(steady_clock::time_point start) {
    auto span = duration_cast<duration<double> >(steady_clock::now() - start);
    return span.count();
}

void printTree(DecisionTree &dt) {
    fstream file("tree.txt", fstream::out);
    for (int i = 0; i < dt.tree.size(); ++i) {
        file << i << " " << dt.tree[i].feature_index << " "
            << dt.tree[i].value << " " << dt.tree[i].label << " "
            << dt.tree[i].left << " " << dt.tree[i].right << endl;
    }
    file.close();
}

void growingForest(vector<DecisionTree> &forest) {
    thread *t = new thread[FOREST_SIZE];

    for (int i = 0; i < forest.size(); ++i) {
        // forest[i].training(USED_FOR_TRAINING, i);
        t[i] = thread(&DecisionTree::training, &forest[i], USED_FOR_TRAINING, i);
    }

    for (int i = 0; i < forest.size(); ++i) {
        t[i].join();
    }

    delete []t;
}


void classifyingData(vector<DecisionTree> &forest, vector<Vote> &votes) {
    printTree(forest[0]);
    for (int i = 0; i < forest.size(); ++i) {
        forest[i].classify(votes);
        cout << "tree " << i << ".....ok" << endl;
    }
    stringstream ss;
    ss << "result/" << USED_FOR_TRAINING << "-" << OK_RATIO << "-" << FOREST_SIZE << ".txt";
    string filename = ss.str();

    fstream file(filename.c_str(), fstream::out);

    fstream debug("vote.txt", fstream::out);
    file << "id,label" << endl;
    for (int i = 0; i < votes.size(); ++i) {
        file << i << "," << setprecision(15) << votes[i].positive_count * 1.0 / FOREST_SIZE << endl;

        debug << i << " " << votes[i]. negtive_count << " "
            << votes[i].positive_count << endl;
    }
    file.close();
    debug.close();

}

void validation(vector<DecisionTree> &forest) {
    double right;
    set<int> indices = DecisionTree::getDataRandomly(TRAINING_SET_SIZE / 10);
    for (set<int>::iterator it = indices.begin(); it != indices.end(); ++it) {
        Vote v;
        for (int i = 0; i < forest.size(); ++i) {
            v += forest[i].voteOne(*it);
        }
        int l = v.negtive_count < v.positive_count ? 1 : 0;
        if (l == data_set[*it].label) right++;
    }
    cout << right / indices.size() << endl;
}

int main() {

    vector<DecisionTree> forest(FOREST_SIZE);
    vector<Vote> votes(TESTING_SET_SIZE);

    auto s = start();
    auto t = start();
    cout << "reading training set...\n";
    read("data/train_data.txt", TRAINING_SET_SIZE);
    cout << stop(t) << "s" << endl << endl;

    t = start();
    cout << "training...\n";
    growingForest(forest);
    cout << stop(t) << "s" << endl << endl;

    validation(forest);

    t = start();
    cout << "reading testing set...\n";
    read("data/test_data.txt", TESTING_SET_SIZE);
    cout << stop(t) << "s" << endl << endl;

    t = start();
    cout << "classifying...\n";
    classifyingData(forest, votes);
    cout << stop(t) << "s" << endl << endl;

    cout << "total time is " << stop(s) << "s" << endl;
    return 0;
}

