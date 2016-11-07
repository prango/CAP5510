import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import plot_learning_curve as plc

#get data in useful form
def get_data(filename):
	df = (pd.read_csv(filename, sep = "\t", skiprows = 70, header = None)).transpose()
	df.columns = df.iloc[0]
	df = df[1:]
	df_type = (pd.read_csv(filename, sep = "\t", skiprows = 32, nrows = 1, header = None)).transpose()
	df.dropna(axis = 1, how = 'any', inplace = True)		#too many genes are getting dropped from 2556 to 2269
	df_type.drop(df_type[:1], inplace = True)

	#changing df_type to classes
	healthy = df_type[df_type[0].str.contains('healthy') == True]
	cancer_patients = df_type[df_type[0].str.contains('healthy') != True]
	df_type.replace(to_replace = healthy, value = 0, inplace = True)
	df_type.replace(to_replace = cancer_patients, value = 1, inplace = True)

	X = np.array(df)
	Y = np.array(df_type)
	return X, Y

#train the model on training data
def train_model(clf, X, Y):
	clf.fit(X,Y.ravel())
	return clf

#predict the output from testing data
def predict_output(clf, X):
	Y_pred = clf.predict(X)
	return Y_pred


def main():
	X, Y = get_data('/Users/GodSpeed/Documents/Courses/Bioinformatics/Project/GSE59856_series_matrix.txt')
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

	clf = KNeighborsClassifier(n_neighbors = 3)

	title = 'Learning Curve (KNeighborsClassifier) initial reults'
	cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 42)
	plc.plot_learning_curve(clf, title, X, Y.ravel(), ylim = (0.7,1.01), cv = cv, n_jobs = 2)

	clf = train_model(clf, X_train, Y_train)
	#print cross_val_score(clf, X_test, Y_test, scoring = 'accuracy')
	Y_pred = predict_output(clf, X_test)
	print confusion_matrix(Y_test, Y_pred)

main()

