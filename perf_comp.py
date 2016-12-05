import sys
import get_data
import KNN
import Naive_Bayes_pyV2 as Naive_Bayes
import DecisionTree
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import warnings

""" script to compare the classifiers and feature selection/reduction methods.
remove call to main function from every other script before running this"""

def compare_methods(argv):
	script, dataset = argv
	filename = '/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/Datasets/GSE' + dataset + '_series_matrix.txt'

	#getting data
	X, Y, df, df_type, class_names = get_data.get_data(filename)
	Xtemp = X

	feature_selection = ['Original Features','Supervised variance by mean ratio', 
	'Unsupervised variance by mean ratio', 'Chi Square']	#add pranav's method
	classifiers = ['KNN', 'Naive Bayes', 'Decision Trees']

	a = [[(0,0)] * len(classifiers)] * len(feature_selection)
	performance = np.array(a)
	#print performance
	
	for model in classifiers:
		for feature_method in feature_selection:
			for num_of_dims in range(75, 76):
				fmeasure_expe = []
				for iter in range(0,1):
					X = Xtemp
					if feature_method == 'None':
						X = X
					if feature_method == 'Supervised variance by mean ratio':
						feature_importance_order = KNN.feature_select_var(X, Y, 
							df_type, class_names)
						X = X[:, np.array(feature_importance_order[
							0:num_of_dims])]
					elif feature_method == 'Unsupervised variance by mean ratio':
						feature_importance_order = KNN.feature_select_std(X)
						X = X[:, np.array(feature_importance_order[
							0:num_of_dims])]
					elif feature_method == 'Chi Square':
						X = Naive_Bayes.feature_reduce(X, Y, num_of_dims)
					#elif	#pranav's method

					#splitting data into training and testing set
					X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
						test_size = 0.2)

					if model == 'KNN':
						clf = KNN.train_model(KNeighborsClassifier( 
							n_neighbors = 1), X_train, Y_train)
					elif model == 'Naive Bayes':
						clf = Naive_Bayes.train_model(GaussianNB(), X_train, 
							Y_train)
					elif model == 'Decision Trees':
						clf = DecisionTree.train_model(
							tree.DecisionTreeClassifier(), X_train, Y_train)

					Y_pred = clf.predict(X_test)

					#checking accuracy
					fmeasure = np.around(f1_score(Y_test, Y_pred, average = 
						'weighted'), decimals = 2)
					fmeasure_expe.append(fmeasure)# + fmeasure

				fmeasure = np.mean(fmeasure_expe) * 100
				col = classifiers.index(model)
				row = feature_selection.index(feature_method)
				#print fmeasure, performance[row][col][0]
				
				if fmeasure > performance[row][col][0]:
					performance[row][col][0] = fmeasure
					performance[row][col][1] = num_of_dims

	print performance
	return performance, feature_selection, classifiers

def plot_table(performance, feature_selection, classifiers, normalize = False,
	cmap = plt.cm.Blues):
	#plt.title(title)
    #plt.colorbar()
    plt.figure()
    x_tick_marks = np.arange(len(classifiers))
    y_tick_marks = np.arange(len(feature_selection))
    plt.xticks(x_tick_marks, classifiers)
    plt.yticks(y_tick_marks, feature_selection)

    c = np.array([[0] * performance.shape[1]] * performance.shape[0])  
    for i, j in itertools.product(range(performance.shape[0]), 
    	range(performance.shape[1])):
      if (i - j) == 0:
        c[i][j] = 1
      else:
        c[i][j] = -1

    for i, j in itertools.product(range(performance.shape[0]), range(
    	performance.shape[1])):
        plt.text( j, i, tuple(performance[i, j]), horizontalalignment="center",
          color="black")

    plt.imshow(c, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.ylabel('Feature Selection Methods')
    plt.xlabel('Classification Methods')
    plt.show()

	

def main(argv):
	warnings.filterwarnings("ignore")
	[performance, feature_selection, classifiers] = compare_methods(argv)		#comparing performance
	plot_table(performance, feature_selection, classifiers)


if __name__ == "__main__":
	main(sys.argv)