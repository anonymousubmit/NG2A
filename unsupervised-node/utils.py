from sklearn.model_selection import GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
import numpy as np
from layers import *
import warnings


def k_fold_cross_val(X, y, search=True):
    mean_acc, std_acc = svc_classify(X, y, search)
    print('SVC Accuracy (Std): %.4f (%.4f)' %(mean_acc, std_acc))
    
    # mean_acc, std_acc = linearsvc_classify(X, y, search)
    # print('LinearSVC Accuracy (Std): %.3f (%.3f)' %(mean_acc, std_acc))

    # mean_acc, std_acc = logisticregression_classify(X, y, search)
    # print('Logistic Regression Accuracy (Std): %.3f (%.3f)' %(mean_acc, std_acc))

    # mean_acc, std_acc = randomforest_classify(X, y, search)
    # print('Random Forest Accuracy (Std): %.3f (%.3f)' %(mean_acc, std_acc))


def svc_classify(x, y, search=True):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []

    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0, n_jobs=8)
        else:
            classifier = SVC(C=10)
            
        classifier.fit(x_train, y_train)

        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        
    return np.mean(accuracies), np.std(accuracies)


def linearsvc_classify(x, y, search=True):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []

    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0, n_jobs=8)
        else:
            classifier = LinearSVC(C=10)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies), np.std(accuracies)


def logisticregression_classify(x, y, search=True):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []

    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LogisticRegression(), params, cv=5, scoring='accuracy', verbose=0, n_jobs=8)
        else:
            classifier = LogisticRegression(C=10)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies), np.std(accuracies)


def randomforest_classify(x, y, search=True):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []

    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0, n_jobs=8)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    
    return np.mean(accuracies), np.std(accuracies)


def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if not hasattr(data, 'val_mask'):
        data.train_mask = data.val_mask = data.test_mask = None

        for i in range(20):
            labels = data.y.numpy()
            val_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            val_index = perm[test_size:test_size + val_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            val_mask = torch.tensor(np.in1d(data_index, val_index), dtype=torch.bool)
            train_mask = ~(val_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            val_mask = val_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if 'train_mask' not in data:
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask
            else:
                data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
                data.val_mask = torch.cat((data.val_mask, val_mask), dim=0)
                data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    else: 
        # in the case of WikiCS
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T

    return data


def evaluate_node_classification_linearmodel(embeddings, data, name, search=True):
    labels = data.y.cpu().detach().numpy()
    emb_dim, num_class = embeddings.shape[1], data.y.unique().shape[0]
    
    test_accs = []

    for i in range(20):
        train_mask = data.train_mask[i].cpu().detach().numpy()
        val_mask = data.val_mask[i].cpu().detach().numpy()

        if name == 'WikiCS':
            test_mask = data.test_mask.cpu().detach().numpy()
        else:
            test_mask = data.test_mask[i].cpu().detach().numpy()

        x_train, y_train = embeddings[train_mask], labels[train_mask]
        x_val, y_val = embeddings[val_mask], labels[val_mask]
        x_test, y_test = embeddings[test_mask], labels[test_mask]

        split_index = [-1] * len(x_train) + [0] * len(x_val)
        X = np.concatenate((x_train, x_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
        pds = PredefinedSplit(test_fold = split_index)

        if search:
            params = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=pds, scoring='accuracy', verbose=0, n_jobs=8)
        else:
            classifier = LinearSVC(C=10)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(X, y)
        
        test_accs.append(accuracy_score(y_test, classifier.predict(x_test)))

    test_acc, test_std = np.mean(test_accs), np.std(test_accs)

    print('\nEvaluate node classification results:')
    print('** Test: {:.4f} ({:.4f}) **\n'.format(test_acc, test_std))


def evaluate_node_classification_LR_model(embeddings, data, name, search=True):
    labels = data.y.cpu().detach().numpy()
    emb_dim, num_class = embeddings.shape[1], data.y.unique().shape[0]
    
    test_accs = []

    for i in range(20):
        train_mask = data.train_mask[i].cpu().detach().numpy()
        val_mask = data.val_mask[i].cpu().detach().numpy()

        if name == 'WikiCS':
            test_mask = data.test_mask.cpu().detach().numpy()
        else:
            test_mask = data.test_mask[i].cpu().detach().numpy()

        x_train, y_train = embeddings[train_mask], labels[train_mask]
        x_val, y_val = embeddings[val_mask], labels[val_mask]
        x_test, y_test = embeddings[test_mask], labels[test_mask]

        split_index = [-1] * len(x_train) + [0] * len(x_val)
        X = np.concatenate((x_train, x_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
        pds = PredefinedSplit(test_fold = split_index)

        if search:
            params = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000), params, cv=pds, scoring='accuracy', verbose=0, n_jobs=8)
        else:
            classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(X, y)
        
        test_accs.append(accuracy_score(y_test, classifier.predict(x_test)))

    test_acc, test_std = np.mean(test_accs), np.std(test_accs)

    print('\nEvaluate node classification results:')
    print('** Test: {:.4f} ({:.4f}) **\n'.format(test_acc, test_std))


def evaluate_node_classification_RF_model(embeddings, data, name, search=True):
    labels = data.y.cpu().detach().numpy()
    emb_dim, num_class = embeddings.shape[1], data.y.unique().shape[0]
    
    test_accs = []

    for i in range(20):
        train_mask = data.train_mask[i].cpu().detach().numpy()
        val_mask = data.val_mask[i].cpu().detach().numpy()

        if name == 'WikiCS':
            test_mask = data.test_mask.cpu().detach().numpy()
        else:
            test_mask = data.test_mask[i].cpu().detach().numpy()

        x_train, y_train = embeddings[train_mask], labels[train_mask]
        x_val, y_val = embeddings[val_mask], labels[val_mask]
        x_test, y_test = embeddings[test_mask], labels[test_mask]

        split_index = [-1] * len(x_train) + [0] * len(x_val)
        X = np.concatenate((x_train, x_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
        pds = PredefinedSplit(test_fold = split_index)

        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=pds, scoring='accuracy', verbose=0, n_jobs=8)
        else:
            classifier = RandomForestClassifier()
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(X, y)
        
        test_accs.append(accuracy_score(y_test, classifier.predict(x_test)))

    test_acc, test_std = np.mean(test_accs), np.std(test_accs)

    print('\nEvaluate node classification results:')
    print('** Test: {:.4f} ({:.4f}) **\n'.format(test_acc, test_std))


