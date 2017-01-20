import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


"""This function is to extract the unprocessed data from the datasets
 and process it into meaningful form"""

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
	df = df[1:
	df.dropna(axis = 1, how = 'all', inplace = True)

	"""dropping non-classifiable data such as post-surgery data, since we don't 
	know whether cancer was found in the later diagnosis or not
	dropping liver from 59856 as only 2 samples exist"""

	data_to_drop = df_type[df_type['!Sample_title'].str.contains
		('post-surgery|ectopic|pr:|liver|cd10cd19|cd34|sjdown|sjball|sjinf|sjmll')
		 == True].axes	
	indices_to_drop = data_to_drop[0].values
	updated_indices = np.apply_along_axis(lambda x: x - 1, 0, indices_to_drop)
	df_type.drop(df_type.index[updated_indices], inplace = True)
	df.drop(df.index[updated_indices], inplace = True)

	#changing df_type to classes
	classes = np.array(['normal', 'pbmc_malignant', 'pbmc_benign',
		'lung cancer', 'br:', 'cns:', 'co:', 'le:', 'me:', 'lc:',
		'ov:', 're:', 'pancreatic','biliary tract','healthy', 'control', 
		'colon', 'stomach','esophagus','sjphall','sje2a','sjhyper',
		'sjerg','sjhypo','sjtall','sjetv'])
	classes_to_names = {'normal':'Normal Patient', 'pbmc_malignant':
	'Malignant Breast Cancer', 'pbmc_benign':'Benign Breast Cancer',
	'lung cancer':'Lung cancer', 'br:':'Breast', 'cns:':
	'Central Nervous System', 'co:':'Colon','le:':'Leukamia', 'me:':'Melanoma',
	'lc:':'Non-small cell Lung', 'ov:':'Ovarian', 're:':'Renal','pancreatic':
	'Pancreatic', 'biliary tract':'Biliary Tract', 'healthy':'Normal Patient',
	'control':'Normal Patient', 'colon':'Colon', 'stomach':'Stomach', 
	'esophagus':'Esophagus', 'sjphall':'PH', 'sje2a':'TCF3-PBX1', 
	'sjhyper':'hyperdiploid', 'sjerg':'MLL', 'sjhypo':'hypodiploid',
	'sjtall':'T-ALL', 'sjetv':'ETV6_RUNX1'}
	class_names_final = []
	num_of_classes = len(classes)
	for i in range(num_of_classes):
		#try:
		class_find = df_type[df_type['!Sample_title'].str.contains
		(classes[i]) == True]
		print class_find.size
		if class_find.size != 0:
			df_type.replace(to_replace = class_find, value = classes_to_names[classes[i]], 
				inplace = True)
			class_names_final.append(classes_to_names[classes[i]])

	#filling values for Nans on the basis of mean of corresponding feature
	imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
	imp.fit(df)
	X = imp.transform(df)
	df = pd.DataFrame(df)
	df.dropna(axis = 1, how = 'any', inplace = True)
	X = np.array(df)
	#print type(df)
	Y = np.array(df_type)
	return X, Y, df, df_type, class_names_final