#include "decisionTree.h"

using namespace std;

clock_t start() {
    return clock();
}

int stop(clock_t start) {
    clock_t finish = clock();
    return (double) (finish - start) / CLOCKS_PER_SEC;
}

int main() {
	srand(time(NULL));
	clock_t t;

    DecisionTree tree;
	
	t = start();
	cout << "reading training set...\n";
    tree.readTrainData("train_data.txt");
    cout << stop(t) << "s" << endl << endl;

	t = start();
	cout << "training...\n";
    tree.training();
    cout << stop(t) << "s" << endl << endl;

    tree.print();

    t = start();
    cout << "reading testing set...\n";
    tree.readTestData("test_data.txt");
    cout << stop(t) << "s" << endl << endl;

    t = start();
    cout << "classifying...\n";
    tree.classify();
    cout << stop(t) << "s" << endl << endl;

//	system("pause");
    return 0;
}
