import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.preprocessing import Imputer
import plot_learning_curve as plc
import plot_confusion_matrix as pcm
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates, radviz
import warnings

#using data sets 32474, 19804, 27562, 59856


#get data in useful form
def get_data(filename):
	#taking row_nums of samples and samples' classes
	with open(filename) as myFile:
		for num, line in enumerate(myFile, 1):
			if '!Sample_title' in line:
				classes_row = num
			if '!series_matrix_table_begin' in line:
				samples_begin = num

	#print classes_row
    #taking samples' classes
	df_type = (pd.read_csv(filename, sep = "\t", skiprows = classes_row - 1,
		 nrows = 1, header = None)).transpose()
	df_type.columns = df_type.iloc[0]
	df_type = df_type[1:]
	#df_type.drop(df_type[:1], inplace = True)
	#print type(df_type[0])
	df_type['!Sample_title'] = df_type['!Sample_title'].str.lower()

	#taking samples
	df = (pd.read_csv(filename, sep = "\t", skiprows = samples_begin + 1,
		 header = None)).transpose()
	df.columns = df.iloc[0]
	df = df[1:]
	df.dropna(axis = 1, how = 'all', inplace = True)

	#dropping non-classifiable data such as post-surgery data, since we don't 
	#know whether cancer was found in the later diagnosis or not
	#dropping liver from 59856 as only 2 samples exist
	data_to_drop = df_type[df_type['!Sample_title'].str.contains
		('post-surgery|ectopic|pr:|liver') == True].axes	
	indices_to_drop = data_to_drop[0].values
	updated_indices = np.apply_along_axis(lambda x: x - 1, 0, indices_to_drop)
	df_type.drop(df_type.index[updated_indices], inplace = True)
	df.drop(df.index[updated_indices], inplace = True)

	#changing df_type to classes
	classes = np.array(['normal', 'pbmc_malignant', 'pbmc_benign',
		'lung cancer', 'br:', 'cns:', 'co:', 'le:', 'me:', 'lc:',
		'ov:', 're:', 'pancreatic','biliary tract','healthy', 'control', 
		'colon', 'stomach','esophagus'])
	classes_to_names = {'normal':'Normal Patient', 'pbmc_malignant':
	'Malignant Breast Cancer', 'pbmc_benign':'Benign Breast Cancer',
	'lung cancer':'Lung cancer', 'br:':'Breast', 'cns:':
	'Central Nervous System', 'co:':'Colon','le:':'Leukamia', 'me:':'Melanoma',
	'lc:':'Non-small cell Lung', 'ov:':'Ovarian', 're:':'Renal','pancreatic':
	'Pancreatic', 'biliary tract':'Biliary Tract', 'healthy':'Normal Patient',
	'control':'Normal Patient', 'colon':'Colon', 'stomach':'Stomach', 
	'esophagus':'Esophagus'}
	class_names_final = []
	num_of_classes = len(classes)
	for i in range(0,num_of_classes):
		#try:
		class_find = df_type[df_type['!Sample_title'].str.contains
		(classes[i]) == True]
		#print class_find.size
		if class_find.size != 0:
			df_type.replace(to_replace = class_find, value = classes_to_names[classes[i]], 
				inplace = True)
			class_names_final.append(classes_to_names[classes[i]])

	#filling values for Nans on the basis of mean of corresponding feature
	#imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
	#imp.fit(df)
	#X = imp.transform(df)
	#df = pd.DataFrame(df)
	df.dropna(axis = 1, how = 'any', inplace = True)
	X = np.array(df)
	#print type(df)
	Y = np.array(df_type)
	return X, Y, df, df_type, class_names_final

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
	print dv
	
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
def feature_select_var(df_type, X, Y):						#univariate supervised since it looks at features in isolation
	samples_healthy = np.where(Y == 'Healthy')[0]
	samples_cancer = np.where(Y == 'Cancer')[0]
	variance_healthy = np.var(X[samples_healthy], axis = 0)
	variance_cancer = np.var(X[samples_cancer], axis = 0)
	mean_healthy = np.mean(X[samples_healthy], axis = 0)
	mean_cancer = np.var(X[samples_cancer], axis = 0)
	feature_importance = pd.DataFrame((variance_healthy + variance_cancer)
		/abs(mean_healthy - mean_cancer))			#the lower, the better
	feature_importance.sort_values(0, axis = 0, ascending = True, 
		inplace = True)
	feature_importance_order = pd.DataFrame(feature_importance.axes)
	feature_importance_order = np.array(feature_importance_order.loc[0,:])
	feature_importance_order = feature_importance_order.astype(int)
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
	X, Y, df, df_type, class_names = get_data(
		'/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/Datasets/GSE59856_series_matrix.txt')
	#print df.head()
	df.dropna(axis = 1, how = 'any', inplace = True)
	#print df_type
	#print type(X)

	visualize_data(X, df, df_type)

	feature_importance_order = feature_select_var(df_type, X, Y)

	feature_importance_order = feature_select_std(X)
	X = X[:,np.array(feature_importance_order[0:200])]	#selecting only top 200 important features

	#print type(X)
	#print np.shape(X)
	
	#print type(feature_importance_order)#[2,1:5]
	#print X[:,select_features]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

	clf = KNeighborsClassifier( n_neighbors = 3)

	title = 'Learning Curve (KNeighborsClassifier) initial results'
	#cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 42)
	#plc.plot_learning_curve(clf, title, X, Y.ravel(), ylim = (0.1,1.01), 
#			cv = cv, n_jobs = 1, scoring = 'accuracy')

	trained_clf = train_model(clf, X_train, Y_train)
	Y_pred = predict_output(trained_clf, X)
	pcm.main(Y, Y_pred, class_names)
	#print cross_val_score(clf, X, Y.ravel(), cv = cv)
	print accuracy_score(Y, Y_pred, normalize = True)
	print np.around(trained_clf.score(X, Y), decimals = 2)
	#print trained_clf.kneighbors(X, n_neighbors = 3)
	#print clf.get_params()

main()