import numpy as np
import argparse

from helper import SVM_classifier

def getLabelsAndData(path, dataset):
    # Read from CSV
    data = np.genfromtxt(f'{path}/{dataset}/fall_features.csv', delimiter=',')
    data2 = np.genfromtxt(f'{path}/{dataset}/not_fall_features.csv', delimiter=',')

    fallLabel = np.ones((len(data),1))
    notFallLabel = np.zeros((len(data2),1))

    # Combine fall and not fall data
    data = np.vstack((data, data2))
    labels = np.vstack((fallLabel, notFallLabel))
    all = np.hstack((data, labels))

    # Shuffle the data randomly
    rng = np.random.default_rng()
    rng.shuffle(all)

    data = all[:,:-1]
    labels = all[:,-1]

    return data, labels

def main(args):
    path = args['source']

    # Get train data and labels
    train_data, train_labels = getLabelsAndData(path=path, dataset='train')

    # Train classifier
    print('Starting training...')
    clf = SVM_classifier(kernel='rbf', gamma='scale', C=0.2)

    clf.crossFoldValidation(train_data, train_labels, cv=10)
    clf.train(train_data, train_labels)

    # Save classifier to file
    clf.saveSVMToFile(args['save_path'])

    # Show results on training data
    print('Train Predicting...')
    train_predictions = clf.predict(train_data)
    clf.results(train_labels, train_predictions, train_data, name='Training')
    clf.plotDecisionBoundary(train_data, train_predictions)

    # Show results on test data
    print('Test Predicting...')
    test_data, test_labels = getLabelsAndData(path=path, dataset='test')
    test_predictions = clf.predict(test_data)
    clf.results(test_labels, test_predictions, test_data, name='Test')
    clf.plotDecisionBoundary(test_data, test_predictions)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source', required=False, default='../data/processed',
                    help='Source data folder for train, val, test features')
    ap.add_argument('--save_path', required=True, default='',
                    help='Path to save classier and pca weights/values')

    args = vars(ap.parse_args())
    main(args)