import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from tree_olnp.tree_olnp import tree_olnp

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

# create test and train sets
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# normalization
sc = StandardScaler()
X_train[:,2:] = sc.fit_transform(X_train[:,2:])
X_test[:,2:] = sc.transform (X_test[:,2:])

# create monte carlo tests
n_aug_samples = 150000
tfprs =      [5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_inits = [ 1e3,  1e3,  1e3, 5e2, 2e2, 2e2, 2e2, 2e2, 2e2, 2e2, 2e2, 2e2]
MC = 10
fpr_test = np.zeros((MC, len(tfprs)))
tpr_test = np.zeros((MC, len(tfprs)))
fpr_train = np.zeros((MC, len(tfprs)))
tpr_train = np.zeros((MC, len(tfprs)))

# run multiple times
for i in range(0, len(tfprs)):
    for j in range(0, MC):

        # classifier definition
        # Note that cross validation is not applied here, it will be implemented in the future versions
        TreeOlnp = tree_olnp(tfpr_ = tfprs[i], tree_depth_ = 5, beta_init_= beta_inits[i], sigmoid_h_ = -2, node_loss_constant_ = 1, projection_type_ = 'manual', max_row_=240, max_col_=360)

        # training
        TreeOlnp.fit(X_train, y_train, n_samples_augmented_min=n_aug_samples)

        # prediction
        y_pred_test = TreeOlnp.predict(X_test)
        # evaluation
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        FPR = fp/(fp+tn)
        TPR = tp/(tp+fn)
        print("Test, Tree-OLNP, TPR: {:.3f}, FPR: {:.3f}".format(TPR, FPR))
        # save outputs
        fpr_test[j, i] = FPR
        tpr_test[j, i] = TPR

        # prediction
        y_pred_train = TreeOlnp.predict(X_train)
        # evaluation
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()
        FPR = fp/(fp+tn)
        TPR = tp/(tp+fn)
        print("({},{}) - Train, Tree-OLNP, TPR: {:.3f}, FPR: {:.3f}".format(i, j, TPR, FPR))
        # save outputs
        fpr_train[j, i] = FPR
        tpr_train[j, i] = TPR

np.savetxt("./output/fpr_test.csv", fpr_test, delimiter=",")
np.savetxt("./output/tpr_test.csv", tpr_test, delimiter=",")
np.savetxt("./output/fpr_train.csv", fpr_train, delimiter=",")
np.savetxt("./output/tpr_train.csv", tpr_train, delimiter=",")