import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import get_data
import plot_learning_curve as plc
import plot_confusion_matrix as pcm
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.feature_selection import chi2


#Aditya: function added
def feature_reduce(X,Y,num_features_to_keep):
    #use the chi-squared method to reduce features and reshape data
    test = SelectKBest(score_func=chi2, k=num_features_to_keep)
    fit = test.fit(X,Y)
    
    #return the data with reduced features 
    return fit.transform(X)
    
#train the model on training data
def train_model(clf, X, Y):
    clf.fit(X,Y.ravel()) 
    return clf

#predict the output from testing data
def predict_output(clf, X):
    Y_pred = clf.predict(X)
    return Y_pred


def main():
    warnings.filterwarnings("ignore")
    #get data in useful form [samples X features]
    X, Y, df, df_type, class_names = get_data.get_data(
        '/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/Datasets/GSE33315_series_matrix.txt')
    
    df.dropna(axis = 1, how = 'any', inplace = True)
   
    #call to reduce features
    X=feature_reduce(X,Y, 200)
    
    #train test splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    
    #defining classifier
    gnb = GaussianNB()
    trained_clf= train_model(gnb,X_train,Y_train)
    

    title = 'Learning Curve (KNeighborsClassifier) initial results'

    Y_pred = predict_output(trained_clf, X_test)


    print accuracy_score(Y_test, Y_pred, normalize = True)
    print np.around(trained_clf.score(X, Y), decimals = 2)
    #print trained_clf.kneighbors(X, n_neighbors = 3)
    #print clf.get_params()

#main()