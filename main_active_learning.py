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
target_FPR = 0.2

# every 5000 sample, calculate TPR and FPR of the transient model using test set
# this application is specific to active learning to better show performance of the model
test_freq = 500

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

# plot transient test performances
f,ax = plt.subplots(1,2,figsize=(8,12))
ax[0].plot(TreeOlnp.test_array_indices_, TreeOlnp.fpr_test_array_, label="FPR")
ax[0].set_xlabel("Number of Samples")
ax[0].set_ylabel("TPR")
ax[0].grid()
ax[0].legend()

ax[1].plot(TreeOlnp.test_array_indices_, TreeOlnp.tpr_test_array_, label="TPR")
ax[1].set_xlabel("Number of Samples")
ax[1].set_ylabel("TPR")
ax[1].grid()
ax[1].legend()
f.savefig('./figures/transient_test_performances.png')