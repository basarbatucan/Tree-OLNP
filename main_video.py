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

# main 
# tree-olnp works for 1,-1 classification
# we expect data to be in tabular form with the latest column as target (check ./data/banana.csv)
data = pd.read_csv('./data/ucsdped2.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# train test split
# add time based for video
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# normalization
sc = StandardScaler()
X_train[:,2:] = sc.fit_transform(X_train[:,2:])
X_test[:,2:] = sc.transform (X_test[:,2:])

# define hyperparameters
parameters = {
    'eta_init': [0.01],               # default, 0.01
    'beta_init': [100],               # default, 100
    'sigmoid_h': [-1],                # default, -1
    'Lambda':[0],                     # default, 0
    'tree_depth':[2],                 # default, 2
    'split_prob':[0.5],               # default, 0.5
    'node_loss_constant':[1]          # default, 1
    }

# classifier definition
# Note that cross validation is not applied here, it will be implemented in the future versions
TreeOlnp = tree_olnp(tfpr = target_FPR, projection_type = 'manual', max_row = 240, max_col = 360)

# hyperparameter tuning
clf = GridSearchCV(TreeOlnp, parameters, verbose=3, cv=2, n_jobs=-1)

# training
clf.fit(X_train, y_train)

# print best params
print(clf.best_params_)

# get best estimator
best_tree_olnp = clf.best_estimator_

# plot space partition
best_tree_olnp.test_init_partitioner(X_test)

# prediction, test
y_pred = best_tree_olnp.predict(X_test)

# evaluation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
FPR = fp/(fp+tn)
TPR = tp/(tp+fn)
print("Test, Tree-OLNP, TPR: {:.3f}, FPR: {:.3f}".format(TPR, FPR))

# plot transient performances
f,ax = plt.subplots(2,2,figsize=(8,12))
ax[0, 0].plot(best_tree_olnp.tpr_train_array_, label="TPR")
ax[0, 0].set_xlabel("Number of Samples")
ax[0, 0].set_ylabel("TPR")
ax[0, 0].grid()
ax[0, 0].legend()

ax[1, 0].plot(best_tree_olnp.fpr_train_array_, label="FPR")
ax[1, 0].set_xlabel("Number of Samples")
ax[1, 0].set_ylabel("FPR")
ax[1, 0].grid()
ax[1, 0].legend()

ax[0, 1].plot(best_tree_olnp.neg_class_weight_train_array_, label="Negative Class weight")
ax[0, 1].plot(best_tree_olnp.pos_class_weight_train_array_, label="Positive Class weight")
ax[0, 1].set_xlabel("Number of Samples")
ax[0, 1].set_ylabel("Class Respective Weights")
ax[0, 1].grid()
ax[0, 1].legend()

for i in range(0, best_tree_olnp.tree_depth):
    scatter_y = best_tree_olnp.mu_train_array_[:,i]
    scatter_x = list(range(0, scatter_y.shape[0]))
    ax[1, 1].scatter(x=scatter_x, y=scatter_y, label="Depth {}".format(i))
ax[1, 1].legend()
ax[1, 1].set_xlabel("Number of Samples")
ax[1, 1].set_ylabel("Expert Weights")
f.savefig('./figures/video_transient_performances.png')