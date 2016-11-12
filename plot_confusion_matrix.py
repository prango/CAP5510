import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, 
  title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #cm = confusion_matrix(Y_test, Y_pred);
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around((cm.astype('float') / cm.sum(axis=1)) * 100, decimals = 2)
        print("Percentage confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    c = np.array([[0] * cm.shape[0]] * cm.shape[1])    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      if (i - j) == 0:
        c[i][j] = 1
      else:
        c[i][j] = -1

    #thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text( j, i, cm[i, j], horizontalalignment="center",
          color="white" if c[i, j] == 1 else "black")

    plt.imshow(c, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
def main(Y_test, Y_pred, class_names):
  cnf_matrix = confusion_matrix(Y_test, Y_pred)
  np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names,
    title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    title='Percentage confusion matrix')

  plt.show()