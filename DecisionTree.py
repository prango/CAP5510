"""This is an implementaion of Decision Tree, before which we are analyzing 
the data also in the same script"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_selection import SelectFromModel 
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.feature_selection import chi2 , mutual_info_classif
import get_data
import plot_learning_curve as plc
import plot_confusion_matrix as pcm
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates, radviz
import warnings

#using data sets 32474, 19804, 27562, 59856, 33315
#Feature reduction with Chi2
def feature_reduce_chi2(X,Y,num_features_to_keep):
    #use the chi-squared method to reduce features and reshape data
    test = SelectKBest(score_func=chi2, k=num_features_to_keep)
    fit = test.fit(X,Y)
    
    #return the data with reduced features 
    return fit.transform(X)

#Feature reductiion with mutual info classifier
def feature_reduce_mic(X,Y,num_features_to_keep):
    
    test = SelectKBest(score_func= mutual_info_classif, k=num_features_to_keep)
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
		'/Users/Pranav/Google Drive/FALL 2016 SUBS/BIO-INFO/Project/data/GSE19804_series_matrix.txt')

	#print class_names
	#print np.shape(X)
	X = normalize(X, axis = 1, copy = False)
	#print X
	X=feature_reduce_chi2(X,Y,100)
	#X= feature_reduce_mic(X,Y,100)

	#visualize_data(X, df, df_type)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)



	clf = tree.DecisionTreeClassifier( )

	#title = 'Learning Curve (Decision Tree initial results)'
	#cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 42)
	#plc.plot_learning_curve(clf, title, X, Y.ravel(), ylim = (0.1,1.01), 
	#		cv = cv, n_jobs = 1, scoring = 'accuracy')


	trained_clf = train_model(clf, X_train, Y_train)
	Y_pred = predict_output(trained_clf, X_test)
	#pcm.main(Y_test, Y_pred, class_names)

	#print accuracy_score(Y, Y_pred, normalize = True)
	#print np.around(trained_clf.score(X, Y), decimals = 2)
	print "F1-score of classification is ", np.around(f1_score(Y_test, Y_pred, average = 'weighted'), decimals = 2)
	#print trained_clf.kneighbors(X, n_neighbors = 3)
	#print clf.get_params()

main()
