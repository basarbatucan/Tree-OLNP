import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from tree_olnp_parallel.tree_olnp_parallel import tree_olnp_parallel
from tree_partition.tree_partition import TreeNode, create_tree, add_indices

# main 

# static
#train = pd.read_csv('./data/train_img_features.csv')
#test = pd.read_csv('./data/test_ucsdped2_img_features.csv')

# dynamic
#train = pd.read_csv('./data/train_dynamic_img_features.csv')
#test = pd.read_csv('./data/test_ucsdped2_dyn_img_features.csv')

# cvpr
train = pd.read_csv('./data/train_ucsdped2_features_cvpr.csv')
test = pd.read_csv('./data/test_ucsdped2_features_cvpr.csv')

# select how many partitions to show
N = 1

# create test and train sets
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
TreeOlnp = tree_olnp_parallel(tfpr_ = 0.1, tree_depth_ = 3, beta_init_= 200, sigmoid_h_ = -2, node_loss_constant_ = 2, projection_type_ = 'manual', max_row_=240, max_col_=360)

# training
TreeOlnp.fit(X_train, y_train)

# prediction, test
y_pred_test = TreeOlnp.predict(X_test)
# evaluation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
FPR = fp/(fp+tn)
TPR = tp/(tp+fn)
print("Test, Tree-OLNP, TPR: {:.3f}, FPR: {:.3f}".format(TPR, FPR))

# prediction, train
y_pred_train = TreeOlnp.predict(X_train)
# evaluation
tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()
FPR = fp/(fp+tn)
TPR = tp/(tp+fn)
print("Train, Tree-OLNP, TPR: {:.3f}, FPR: {:.3f}".format(TPR, FPR))

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

# take best N partition
best_partition_indices = np.argsort(TreeOlnp.expert_weights_[-1,:])
best_partition_indices = best_partition_indices[:N]
# add loss of the simplest expert fr comparison
if not 0 in best_partition_indices:
    best_partition_indices = np.append(best_partition_indices, 0)
if not len(TreeOlnp.partitions_)-1 in best_partition_indices:
    best_partition_indices = np.append(best_partition_indices, len(TreeOlnp.partitions_)-1)

# plot expert weights
f,ax = plt.subplots(1,1,figsize=(8,12))
for i in range(0, len(best_partition_indices)):
    best_partition_index = best_partition_indices[i]
    expert_w = TreeOlnp.expert_weights_[:,best_partition_index]
    ax.plot(expert_w, label=str(TreeOlnp.partitions_[best_partition_index]))
ax.legend()
ax.set_yscale('log')
ax.set_xlabel("Number of Samples")
ax.set_ylabel("Avg expert loss per sample")
f.savefig('./figures/partition_weights.png')