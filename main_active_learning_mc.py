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
MC = 2

#data_name = 'banana'
#data_name = 'telescope'
#data_name = 'miniboone'
data_name = 'codrna'

#file_name = 'random'
file_name = 'active'

active_learning_ = True
save_file_name = file_name + '_' + data_name

# total number of tests
total_tests = 100

# main 
# np-nn works for 1,-1 classification
# we expect data to be in tabular form with the latest column as target (check ./data/banana.csv)
data = pd.read_csv('./data/{}.csv'.format(data_name))
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

# get n_samples
if X_train.shape[0] > 150e3:
    test_freq = int(X_train.shape[0]/total_tests)
else:
    test_freq =  int(max(150e3, X_train.shape[0]*2)/total_tests)

# create datasets
indices = np.ones((MC, total_tests*2))*-1
tpr = np.ones((MC, total_tests*2))*-1
NP = np.ones((MC, total_tests*2))*-1

# start running MC
for i in range(0, MC):
    # print progress
    print('MC: {}'.format(i+1))

    # classifier definition
    # Note that cross validation is not applied here, it will be implemented in the future versions
    TreeOlnp = tree_olnp(
        tfpr = target_FPR, 
        projection_type = 'iterative_PCA', 
        tree_depth=5, 
        active_learning=active_learning_, 
        exploration_prob=0.3, 
        uncertainity_threshold=0.5)

    # training with test
    TreeOlnp.fit(X_train, y_train, X_test=X_test, y_test=y_test, test_freq=test_freq)

    # get results
    indices_ = TreeOlnp.test_array_indices_
    tpr_ = TreeOlnp.tpr_test_array_
    NP_ = TreeOlnp.np_test_array_

    # train size
    train_size = indices_.shape[0]

    # save the output
    indices[i, :train_size] = indices_
    tpr[i, :train_size] = tpr_
    NP[i, :train_size] = NP_

np.savetxt("./output/active/{}_indices.csv".format(save_file_name), indices, delimiter=",")
np.savetxt("./output/active/{}_tpr.csv".format(save_file_name), tpr, delimiter=",")
np.savetxt("./output/active/{}_np.csv".format(save_file_name), NP, delimiter=",")
