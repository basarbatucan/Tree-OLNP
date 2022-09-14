import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from tree_olnp.tree_olnp import tree_olnp

# main 
# np-nn works for 1,-1 classification
# we expect data to be in tabular form with the latest column as target (check ./data/banana.csv)
#data = pd.read_csv('./data/ucsdped2.csv')
#X = data.iloc[:,:-1].values
#y = data.iloc[:,-1].values

# train test split
# add time based for video
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train = pd.read_csv('./data/train_img_features.csv')
test = pd.read_csv('./data/test_ucsdped2_img_features.csv')

#train = pd.read_csv('./data/train_dynamic_img_features.csv')
#test = pd.read_csv('./data/test_ucsdped2_dyn_img_features.csv')

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# normalization
sc = StandardScaler()
X_train[:,2:] = sc.fit_transform(X_train[:,2:])
X_test[:,2:] = sc.transform (X_test[:,2:])

# classifier definition
# Note that cross validation is not applied here, it will be implemented in the future versions
TreeOlnp = tree_olnp(tfpr_ = 0.1, tree_depth_ = 4, sigmoid_h_ = -2, node_loss_constant_ = 2, projection_type_ = 'manual', max_row_=240, max_col_=360)

# training
TreeOlnp.fit(X_train, y_train)

# prediction
y_pred = TreeOlnp.predict(X_test)

# evaluation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
FPR = fp/(fp+tn)
TPR = tp/(tp+fn)
print("Tree-OLNP, TPR: {:.3f}, FPR: {:.3f}".format(TPR, FPR))

# plot transient performances
f,ax = plt.subplots(2,2,figsize=(8,12))
ax[0, 0].plot(TreeOlnp.tpr_train_array_, label="TPR")
ax[0, 0].set_xlabel("Number of Samples")
ax[0, 0].set_ylabel("TPR")
ax[0, 0].grid()
ax[0, 0].legend()

ax[1, 0].plot(TreeOlnp.fpr_train_array_, label="FPR")
ax[1, 0].set_xlabel("Number of Samples")
ax[1, 0].set_ylabel("FPR")
ax[1, 0].grid()
ax[1, 0].legend()

ax[0, 1].plot(TreeOlnp.neg_class_weight_train_array_, label="Negative Class weight")
ax[0, 1].plot(TreeOlnp.pos_class_weight_train_array_, label="Positive Class weight")
ax[0, 1].set_xlabel("Number of Samples")
ax[0, 1].set_ylabel("Class Respective Weights")
ax[0, 1].grid()
ax[0, 1].legend()

for i in range(0, TreeOlnp.tree_depth_):
    scatter_y = TreeOlnp.mu_train_array_[:,i]
    scatter_x = list(range(0, scatter_y.shape[0]))
    ax[1, 1].scatter(x=scatter_x, y=scatter_y, label="Depth {}".format(i))
ax[1, 1].legend()
ax[1, 1].set_xlabel("Number of Samples")
ax[1, 1].set_ylabel("Expert Weights")
f.savefig('./figures/video_transient_performances.png')