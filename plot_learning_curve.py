import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, title, X, Y,  ylim = None, 
	cv = None, n_jobs = 1, train_sizes=np.linspace(.1, 1.0, 5)):
	
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(model, X, Y, 
		cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
	train_scores_mean = np.mean(train_scores, axis = 1)
	train_scores_std = np.std(train_scores, axis = 1)
	test_scores_mean = np.mean(test_scores, axis = 1)
	test_scores_std = np.std(test_scores, axis = 1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					train_scores_mean + train_scores_std, alpha = 0.1)
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
	plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = 'Training score')
	plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = 'Cross-validation score')

	plt.legend(loc = 'best')
	plt.show()




#title = 'Learning Curve (Logistic regression)'

#cv = ShuffleSplit(n_splits = 200, test_size = 0.3, random_state = 0)

#model = linear_model.LogisticRegression()
#plot_learning_curve(model, title, X_train, Y_train, ylim = (0.7,1.01), cv =cv, n_jobs = -1)
