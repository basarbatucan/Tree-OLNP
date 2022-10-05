from tkinter.tix import Tree
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from tree_olnp.tree_olnp import tree_olnp

# Target False Alarm
# NP framework aims to maximize the detection power while upper bounding the false alarm rate
# target false alarm rate should be determined by the user
target_FPR = 0.1
MC = 10

# every 5000 sample, calculate TPR and FPR of the transient model using test set
# this application is specific to active learning to better show performance of the model
test_freq = 1000

# main 
# np-nn works for 1,-1 classification
# we expect data to be in tabular form with the latest column as target (check ./data/banana.csv)
data = pd.read_csv('./data/banana.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

# get n_samples
n_samples =  int(1.1*max(150e3, X_train.shape[0])/test_freq)

# create datasets
indices = np.empty((MC, n_samples))
fpr = np.empty((MC, n_samples))
tpr = np.empty((MC, n_samples))
for i in range(0, MC):
    # print progress
    print('MC: {}'.format(i+1))

    # classifier definition
    # Note that cross validation is not applied here, it will be implemented in the future versions
    TreeOlnp = tree_olnp(
        tfpr = target_FPR, 
        projection_type = 'iterative_PCA', 
        tree_depth=5, 
        active_learning=False, 
        exploration_prob=0.3, 
        uncertainity_threshold=0.95)

    # training with test
    TreeOlnp.fit(X_train, y_train, X_test=X_test, y_test=y_test, test_freq=test_freq)

    # get results
    indices_ = TreeOlnp.test_array_indices_
    fpr_ = TreeOlnp.fpr_test_array_
    tpr_ = TreeOlnp.tpr_test_array_

    # train size
    train_size = indices_.shape[0]

    # save the output
    indices[i, :train_size] = indices_
    fpr[i, :train_size] = fpr_
    tpr[i, :train_size] = tpr_

np.savetxt("./output/active/random_indices.csv", indices, delimiter=",")
np.savetxt("./output/active/random_fpr.csv", fpr, delimiter=",")
np.savetxt("./output/active/random_tpr.csv", tpr, delimiter=",")