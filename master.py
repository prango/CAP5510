import sys
import get_data
import KNN
import Naive_Bayes_pyV2 as Naive_Bayes
import DecisionTree
import plot_confusion_matrix as pcm
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import f1_score
import plot_learning_curve as plc
import numpy as np
import pandas as pd
import warnings

"""master program to select/reduce/extract features and train a classifiers
remove call to main function from every other script before running this"""

def main(argv):
	warnings.filterwarnings("ignore")
	script, dataset, model, feature_selection, num_of_dims = argv
	num_of_dims = int(num_of_dims)
	filename = '/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/Datasets/GSE' + dataset + '_series_matrix.txt'

	#getting data
	X, Y, df, df_type, class_names = get_data.get_data(filename)

	#feature selection/reduction/extraction
	if feature_selection == 'feature_var':
		feature_importance_order = KNN.feature_select_var(X, Y, df_type, 
			class_names)
		X = X[:, np.array(feature_importance_order[0:num_of_dims])]
	elif feature_selection == 'feature_std':
		feature_importance_order = KNN.feature_select_std(X)
		X = X[:, np.array(feature_importance_order[0:num_of_dims])]
	elif feature_selection == 'chi':
		X = Naive_Bayes.feature_reduce(X, Y, num_of_dims)
	elif feature_selection == 'mutual_info_clf':
		X = DecisionTree.feature_reduce_mic(X, Y, num_of_dims)	#for pranav's method

	
	#splitting data into training and testing set
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

	#training the specified model
	if model == "knn":
		clf = KNN.train_model(KNeighborsClassifier( n_neighbors = 1), X_train, 
			Y_train)
	elif model == 'dt':
		clf = DecisionTree.train_model(tree.DecisionTreeClassifier(), X_train,
			 Y_train)
	elif model == 'nb':
		clf = Naive_Bayes.train_model(GaussianNB(), X_train, Y_train)

	#predict the output
	Y_pred = clf.predict(X_test)
	pcm.main(Y_test, Y_pred, class_names)
	#cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 42)
	#plc.plot_learning_curve(clf, title, X, Y.ravel(), ylim = (0.1,1.01), 
	#		cv = cv, n_jobs = 1, scoring = 'accuracy')
	print "F1-score of classification is ", np.around(f1_score(Y_test, Y_pred, 
		average = 'weighted'), decimals = 2)

	#model.main(dims)


if __name__ == "__main__":
	main(sys.argv)