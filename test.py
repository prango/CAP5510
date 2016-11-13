import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import plot_learning_curve as plc
import plot_confusion_matrix as pcm
import matplotlib.pyplot as plt

#get data in useful form
def get_data(filename):
	#taking row_nums of samples and samples' classes
	with open(filename) as myFile:
		for num, line in enumerate(myFile, 1):
			if '!Sample_title' in line:
				classes_row = num
			if '!series_matrix_table_begin' in line:
				samples_begin = num

    #taking samples' classes
	df_type = (pd.read_csv(filename, sep = "\t", skiprows = classes_row - 1,
		 nrows = 1, header = None)).transpose()
	df_type.drop(df_type[:1], inplace = True)
	df_type[0] = df_type[0].str.lower()

	#taking samples
	df = (pd.read_csv(filename, sep = "\t", skiprows = samples_begin + 1,
		 header = None)).transpose()
	df.columns = df.iloc[0]
	df = df[1:]
	df.dropna(axis = 1, how = 'any', inplace = True)		
	#too many genes are getting dropped from 2556 to 2269 in the above step


	#dropping non-classifiable data such as post-surgery data, since we don't 
	#know whether cancer was found in the later diagnosis or not
	data_to_drop = df_type[df_type[0].str.contains('post-surgery') == True].axes
	indices_to_drop = data_to_drop[0].values
	updated_indices = np.apply_along_axis(lambda x: x - 1, 0, indices_to_drop)
	df_type.drop(df_type.index[updated_indices], inplace = True)
	df.drop(df.index[updated_indices], inplace = True)

	#changing df_type to classes
	healthy = df_type[df_type[0].str.contains('healthy|normal')
		 == True]
	cancer_patients = df_type[df_type[0].str.contains('healthy|normal')
		 != True]
	df_type.replace(to_replace = healthy, value = 0, inplace = True)
	df_type.replace(to_replace = cancer_patients, value = 1, inplace = True)

	X = np.array(df)
	Y = np.array(df_type)
	return X, Y

#visualize data
def visualize(X):
	variance = np.var(X, axis = 0)
	xlength = np.arange(np.shape(variance)[0])
	plt.plot(xlength, variance)
	plt.ylabel('Variance')
	plt.xlabel('Gene probe')
	plt.title('Variance of expression level of the genes')
	plt.show()

#train the model on training data
def train_model(clf, X, Y):
	clf.fit(X,Y.ravel())
	return clf

#predict the output from testing data
def predict_output(clf, X):
	Y_pred = clf.predict(X)
	return Y_pred


def main():
	X, Y = get_data(
		'/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/Datasets/GSE19804_series_matrix.txt')

	visualize(X)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

	clf = KNeighborsClassifier( n_neighbors = 1)

	title = 'Learning Curve (KNeighborsClassifier) initial reults'
	cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 42)
	plc.plot_learning_curve(clf, title, X, Y.ravel(), ylim = (0.1,1.01), 
			cv = cv, n_jobs = 1, scoring = 'accuracy')

	trained_clf = train_model(clf, X_train, Y_train)
	Y_pred = predict_output(trained_clf, X)
	class_names = ['healthy', 'cancer']
#	pcm.main(Y, Y_pred, class_names)
	#print cross_val_score(clf, X, Y.ravel(), cv = cv)
	#print accuracy_score(Y, Y_pred, normalize = True)
	#print np.around(trained_clf.score(X, Y), decimals = 2)
	#print trained_clf.kneighbors(X, n_neighbors = 3)
	#print clf.get_params()

main()

