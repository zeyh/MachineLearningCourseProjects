'''
Project 3 - Due May 8, 2018, submitted on May 9 
CSC420
Zeyang Huang
Lab Partner: Rachel Fu, Ashley Hayes
'''
'''
test train split, scale the data
use SVM Logistic regression KNN and perceptron to classify the breast cancer data
grid search with K-fold cross validation for hyperparameter tunings in SVM and KNN algorithms
print out the accucacy scores
'''
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import train_test_split # for evaluation
from sklearn import metrics # for confusion matrix
from sklearn.datasets import fetch_20newsgroups
from random import shuffle
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def main():
    #load the data
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
##    print(X.shape)
##    print(dataset.DESCR)

    #split test training - 65% training 35% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
##    print(X_train.shape, y_train.shape)#70 percent training set
##    print(X_test.shape, y_test.shape)#30 percent testing set
    
    #scale the training data 
    X_scaled = preprocessing.scale(X_train)

    #plot the data
##    colormap = np.array(['r', 'k'])
##    plt.scatter(X_scaled[0], X_scaled[1], c=colormap[y_train], s=40)
##    plt.show()

    #normalize the testing set seperately
    X_scaled_testing = preprocessing.scale(X_test)

    #names that matched 0 and 1 in y data
    target_names = ["malignant", "benign"]
    
    ## SVM   
    print('****************************************')
    print('Support Vector Machine')
    #get the classifier
    clf = svmtrainingfunction(X_scaled, y_train)
    score = clf.score(X_scaled_testing, y_test)
    print("Final Accuracy Score on testing set : ", score)
    #make predictions     
    predicted = clf.predict(X_scaled_testing)
    #results
    print("report: ")
    print(metrics.classification_report(y_test, predicted, target_names=target_names))
    print("confusion matrix: ")
    print(metrics.confusion_matrix(y_test, predicted))


    ## logistic regression
    print('****************************************')
    print('Logistic Regression')
    #get the classifier
    clf2 = LRtrainingfunction(X_scaled, y_train)
    score2 = clf2.score(X_scaled_testing, y_test)
    print("Final Accuracy Score on testing set : ", score2)
    #make predictions 
    predicted2 = clf2.predict(X_scaled_testing)
    print("report: ")
    print(metrics.classification_report(y_test, predicted2, target_names=target_names))
    print("confusion matrix: ")
    print(metrics.confusion_matrix(y_test, predicted2))

    ## perceptron
    print('****************************************')
    print('Perceptron')
    #get the classifier
    clf3 = perceptronfunction(X_scaled, y_train)
    score3 = clf3.score(X_scaled_testing, y_test)
    print("Final Accuracy Score on testing set : ", score3)
    #make predictions 
    predicted3 = clf3.predict(X_scaled_testing)
    print("report: ")
    print(metrics.classification_report(y_test, predicted3, target_names=target_names))
    print("confusion matrix: ")
    print(metrics.confusion_matrix(y_test, predicted3))
    
    ## K neighbor
    print('****************************************')
    print('k neighbor')
    #get the classifier
    clf4 = kkneighborfunction(X_scaled, y_train)
    score4 = clf4.score(X_scaled_testing, y_test, sample_weight=None)
    print("Final Accuracy Score on testing set : ", score4)
    #make predictions 
    predicted4 = clf4.predict(X_scaled_testing)   
    print("report:")
    print(metrics.classification_report(y_test, predicted4, target_names=target_names))
    print("confusion matrix: ")
    print(metrics.confusion_matrix(y_test, predicted4))



    
#take the scaled x training set and y training set, return the trained classifier                    
def svmtrainingfunction(X, y):
    #all the possible kernals
    kernalset = ["linear","rbf","sigmoid","poly"]
    #possible c value from 0.1 to 2.0
    cset = []
    for i in range (1,21):
        #make the float to one decimal
        cset.append(float("%.1f" % (0.1*i)))

##    #k-fold cross validation for hyperparameter grid search 
##    betterscore = 0
##    betterset = ["",0]
##    numberSplits = 10 
##    kf = KFold(n_splits=numberSplits, random_state=None, shuffle=False)
##    kf.get_n_splits(X)   
##    for train_index, test_index in kf.split(X):
##        X_train, X_test = X[train_index], X[test_index]
##        y_train, y_test = y[train_index], y[test_index]
##        for i in range(len(kernalset)):
##            for i1 in range(len(cset)):
##                clf = svm.SVC(kernel=kernalset[i], C=cset[i1]).fit(X[train_index], y[train_index])
##                score = clf.score(X[test_index], y[test_index])
##                if score >= betterscore:
##                    betterscore = score
##                    betterset[0] = kernalset[i]
##                    betterset[1] = cset[i1] 
##    print(betterset)
##    clf = svm.SVC(kernel=betterset[0], C=betterset[1]).fit(X, y)

    #using the methods from the package 
    parameters = {'kernel':("linear","rbf","sigmoid","poly"), 'C':cset}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, cv=4) #cv could be some other numbers #grid search for the hyperparameters using k-fold
    clf.fit(X,y)
    #print the best hyperparameter set it chose
    clf_params = clf.best_params_
    print("best parameters: {}".format(clf_params))
    
##    #graph the grid search result
##    scores = [x[1] for x in clf.grid_scores_]
##    scores = np.array(scores).reshape(len(kernalset), len(cset))
##    for ind, i in enumerate(kernalset):
##        plt.plot(cset, scores[ind], label='Kernel: ' + str(i))
##    plt.legend()
##    plt.xlabel('hyperparameter C: ')
##    plt.ylabel('Mean score')
##    plt.show()

    print ("Accuracy on the training set: " + str(clf.score(X,y)*100) + "%")
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  
    return clf


#takes the same input as above, returns the Logistic regression classifier 
def LRtrainingfunction(X, y):
    clf2=LogisticRegression()
    
##    numberSplits = 10 #the number of K
##    kf = KFold(n_splits=numberSplits, random_state=None, shuffle=False)    
##    for train_index, test_index in kf.split(X):
##        X_train, X_test = X[train_index], X[test_index]
##        y_train, y_test = y[train_index], y[test_index]
##        clf2.fit(X_train, y_train)
        
    clf2.fit(X, y)
    
    print ("Accuracy on the training set: " + str(clf2.score(X,y)*100) + "%")
    scores = cross_val_score(clf2, X, y, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf2


#takes the same input as above, returns the KNN classifier    
def kkneighborfunction(X,y):
    neigh = KNeighborsClassifier(n_neighbors=5) 
    neigh.fit(X, y) 

    print ("Accuracy on the training set: " + str(neigh.score(X,y)*100) + "%")
    scores = cross_val_score(neigh, X, y, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return neigh


#takes the same input as above, returns the perceptron classifier    
def perceptronfunction(X, y):
    # hyperparamete: max_iteration number, verbose, eta0
    clf = Perceptron(max_iter=1000, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
    clf.fit(X, y)
    
    # Print the results
    print ("Accuracy on the training set: " + str(clf.score(X,y)*100) + "%")
    
##    # drawing: calc the hyperplane | decision boundary
##    colormap = np.array(['r', 'k'])
##    plt.scatter(X[1], X[2], c=colormap[y], s=40)
##    ymin, ymax = plt.ylim()
##    w = clf.coef_[0]
##    a = -w[0] / w[1]
##    xx = np.linspace(ymin, ymax)
##    yy = a * xx - (clf.intercept_[0]) / w[1]
## 
##    # Plot the line
##    plt.plot(yy,xx, 'k-')
####    plt.show()

    #cross validation cv = 5
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return clf


main()

