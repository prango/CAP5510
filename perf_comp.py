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

def compare_methods(dataset):
	filename = '/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/Datasets/GSE' + dataset + '_series_matrix.txt'

	#getting data
	X, Y, df, df_type, class_names = get_data.get_data(filename)
	Xtemp = X

	feature_selection = ['Original Features',
	#'Supervised variance by mean ratio', 
	'Semi-supervised variance by mean ratio',
	'Unsupervised variance by mean ratio', 
	'Chi Square', 'ANOVA F-classifier']
	classifiers = ['KNN', 'Naive Bayes', 'Decision Trees']
	max_features = 500
	f_measure_each = np.array([[0] * int(max_features/100)] * 
		(len(classifiers) * (len(feature_selection) - 1)))
	#print np.shape(f_measure_each)
	#f_measure_each = []
	counter = 0
	method_combo = []

	a = [[(0,0)] * len(classifiers)] * (len(feature_selection))
	#ignoring original features from feature_selection
	performance = np.array(a)
	#print performance
	
	for model in classifiers:
		for feature_method in feature_selection:
			for num_of_dims in range(100, max_features + 1, 100):
				fmeasure_expe = []
				for iter in range(0,5):
					X = Xtemp
					if feature_method == 'Original Features':
						X = X
					elif feature_method == 'Supervised variance by mean ratio':
						feature_importance_order = KNN.feature_select_var(X, Y, 
							df_type, class_names)
						X = X[:, np.array(feature_importance_order[
							0:num_of_dims])]
					elif feature_method == 'Semi-supervised variance by mean ratio':
						feature_importance_order = KNN.feature_std_distance(X, Y, 
							df_type, class_names)
						X = X[:, np.array(feature_importance_order[
							0:num_of_dims])]
					elif feature_method == 'Unsupervised variance by mean ratio':
						feature_importance_order = KNN.feature_select_std(X)
						X = X[:, np.array(feature_importance_order[
							0:num_of_dims])]
					elif feature_method == 'Chi Square':
						X = Naive_Bayes.feature_reduce(X, Y, num_of_dims)
					elif feature_method == 'ANOVA F-classifier':
						X = DecisionTree.feature_reduce_f_class_if(X, Y, num_of_dims)

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
					fmeasure_expe.append(fmeasure)

				fmeasure = np.mean(fmeasure_expe) * 100
				row = feature_selection.index(feature_method)
				col = classifiers.index(model)

				if feature_method != 'Original Features':	#storing f-measure for num_of_dims
					f_measure_each[counter][int(num_of_dims/100) - 1] = fmeasure

				
				if fmeasure > performance[row][col][0]:
					performance[row][col][0] = fmeasure
					if feature_method == 'Original Features':
						performance[row][col][1] = np.shape(X)[1]
					else:
						performance[row][col][1] = num_of_dims

			if feature_method != 'Original Features':
				counter = counter + 1
				method_combo.append(model + '|' + feature_method)

	method_combo = np.array(method_combo)
	print performance
	#print f_measure_each
	f_measure_dims = pd.DataFrame(data = f_measure_each, 
		columns = np.arange(100, max_features + 1, 100),
		index = method_combo)
	print f_measure_dims
	return performance, feature_selection, classifiers, f_measure_dims

def plot_table(performance, feature_selection, classifiers, dataset, 
	f_measure_dims, normalize = False, cmap = plt.cm.Blues):
    plt.figure()
    plt.title("Feature selection method versus classification model for GSE" + 
    	dataset)
    x_tick_marks = np.arange(len(classifiers))
    y_tick_marks = np.arange(len(feature_selection))
    plt.xticks(x_tick_marks, classifiers)
    plt.yticks(y_tick_marks, feature_selection)

    c = np.array([[-1] * performance.shape[1]] * performance.shape[0]) 

    for i, j in itertools.product(range(performance.shape[0]), range(
    	performance.shape[1])):
        plt.text( j, i, tuple(performance[i, j]), horizontalalignment="center",
          color="black")

    plt.imshow(c, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.ylabel('Feature Selection Methods')
    plt.xlabel('Classification Methods')
    plt.show()

    #plt.figure()
    f_measure_dims.plot(colormap = plt.cm.jet)
    plt.title('Performance Rate comparison for Feature Selection Method plus Classifying model GSE' + dataset)
    plt.xlabel('Number of features selected')
    plt.ylabel('Accuracy %')
    #ax.legend_.remove()
    plt.show()

	

def main(argv):
	warnings.filterwarnings("ignore")
	script, dataset = argv
	[performance, feature_selection, classifiers, f_measure_dims] = compare_methods(dataset)		#comparing performance
	plot_table(performance, feature_selection, classifiers, dataset, f_measure_dims.transpose())


if __name__ == "__main__":
	main(sys.argv)