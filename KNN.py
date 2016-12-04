"""This is an implementaion of K nearest neighbors, before which we are analyzing 
the data also in the same script"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import get_data
import plot_learning_curve as plc
import plot_confusion_matrix as pcm
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates, radviz
import warnings

#using data sets 32474, 19804, 27562, 59856, 33315

#visualize data
def visualize_data(X, df, df_type):

	"""checking the variance, mean and standard deviation of features"""
	variance = np.var(X, axis = 0)
	sd = np.std(np.float64(X), axis = 0)
	mn = np.mean(X, axis = 0)
	probe = np.array(df.columns)
	dv = pd.DataFrame(np.array([probe, variance, mn, sd]), 
		index = ['Probe', 'Variance', 'Mean', 'Standard Deviation'])
	dv.sort_values('Variance', axis = 1, ascending = False, inplace = True)
	#print dv
	
	"""visualizing the variance of features"""
	xlength = np.arange(np.shape(variance)[0])
	plt.plot(xlength, variance)
	plt.ylabel('Variance')
	plt.xlabel('Gene probe')
	plt.title('Variance of expression level of the genes')
	plt.show()

	"""visualizing the separatability of 8 top most variant features"""
	#featuresForGraph = np.array(dv.loc[0, :])
	#print dv.head()
	features = dv.loc['Probe',:]
	featuresForGraph = features[0:8]
	plt.figure()
	dfPlot = df.loc[:, featuresForGraph]
	data = pd.concat([dfPlot, df_type], axis = 1)
	parallel_coordinates(data, '!Sample_title', colormap = 'jet')
	plt.title('8 Genes with most variant expression level')
	plt.xlabel('Gene probe')
	plt.ylabel('Variance')
	plt.show()

	"""visualizing the separatability of 8 top most variant features
	using normalized values""" 
	plt.figure();
	radviz(data, '!Sample_title')
	plt.title('8 Genes with most variant expression level')
	#plt.xlabel('Normalized')
	plt.tick_params(axis = 'both', bottom = 'off', left = 'off', 
		labelbottom  = 'off', labelleft = 'off')
	plt.show()

	"""visualizing the seperability of 8 least variant features"""
	features = dv.loc['Probe',:]
	featuresForGraph = features.tail(8)

	plt.figure()
	dfPlot = df.loc[:, featuresForGraph]
	data = pd.concat([dfPlot, df_type], axis = 1)
	parallel_coordinates(data, '!Sample_title', colormap = 'jet')
	plt.title('8 Genes with least variant expression level')
	plt.xlabel('Gene probe')
	plt.ylabel('Variance')
	plt.show()

	"""visualizing the sepratability of 8 least variant features using 
	normalized values"""
	plt.figure();
	radviz(data, '!Sample_title')
	plt.title('8 Genes with least variant expression level')
	plt.tick_params(axis = 'both', bottom = 'off', left = 'off', 
		labelbottom  = 'off', labelleft = 'off')
	plt.show()

#selecting features
def feature_select_var(X, Y, df_type, class_names):						
#univariate supervised since it looks at features in isolation
	samples_type = []
	class_variance = []
	#class_std = []
	class_mean = []
	mean_classes = []
	num_of_classes = np.size(class_names)
	for classes in class_names:
		samples_type = np.where(Y == classes)[0]
		class_variance.append(np.var(X[samples_type], axis = 0))
		#class_std.append(np.std(np.float64(X[samples_type]), axis = 0))
		class_mean.append(np.mean(X[samples_type], axis = 0))
	#print np.shape(class_variance)
	#print class_std
	#print X.shape[1]
	feature_dist = np.zeros(X.shape[1])
	for i in range(num_of_classes):
		feature_dist = np.subtract(feature_dist, class_mean[i][:])
	#print np.shape(feature_dist)
	feature_importance = pd.DataFrame(abs(np.sum(class_variance, axis = 0)/feature_dist))
			#the lower, the better
	feature_importance.sort_values(0, axis = 0, ascending = True, 
		inplace = True)
	feature_importance_order = pd.DataFrame(feature_importance.axes)
	feature_importance_order = np.array(feature_importance_order.loc[0,:])
	feature_importance_order = feature_importance_order.astype(int)
	#print feature_importance_order
	#print X[:,np.array(feature_importance_order[0:5])]
	return feature_importance_order

def feature_select_std(X):					#unsupervised
	mean_features = np.mean(X, axis = 0)
	mean_std = np.std(np.float64(X), axis = 0)
	feature_importance = pd.DataFrame(mean_std / mean_features)		#the higher the better
	feature_importance.sort_values(0, axis = 0, ascending = False, 
		inplace = True)
	feature_importance_order = pd.DataFrame(feature_importance.axes)
	feature_importance_order = np.array(feature_importance_order.loc[0,:])
	feature_importance_order = feature_importance_order.astype(int)
	#print feature_importance
	return feature_importance_order

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
		'/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/Datasets/GSE19804_series_matrix.txt')
	print np.shape(X)
	X = normalize(X, axis = 1, copy = False)
	#print X

	#visualize_data(X, df, df_type)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
	#feature_importance_order = feature_select_var(X_train, Y_train, df_type, class_names)
	#feature_importance_order = feature_select_var(X, Y, df_type, class_names)

	feature_importance_order = feature_select_std(X)
	#X = X[:,np.array(feature_importance_order[0:500])]	#selecting only top 200 important features
	#X_train = X_train[:,np.array(feature_importance_order[0:150])]
	#X_test = X_test[:,np.array(feature_importance_order[0:150])]

	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

	clf = KNeighborsClassifier( n_neighbors = 1)

	title = 'Learning Curve (KNeighborsClassifier) initial results'
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