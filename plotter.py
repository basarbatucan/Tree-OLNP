import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

f,ax = plt.subplots(1,2,figsize=(10,8))

# static
fpr_train_static = pd.read_csv('./output/static/fpr_train.csv', header=None)
tpr_train_static = pd.read_csv('./output/static/tpr_train.csv', header=None)
fpr_train_static_x = np.concatenate([np.array([0]), fpr_train_static.mean(), np.array([1])])
tpr_train_static_y = np.concatenate([np.array([0]), tpr_train_static.mean(), np.array([1])])
auc_static_train = np.trapz(tpr_train_static_y, fpr_train_static_x)

fpr_test_static = pd.read_csv('./output/static/fpr_test.csv', header=None)
tpr_test_static = pd.read_csv('./output/static/tpr_test.csv', header=None)
fpr_test_static_x = np.concatenate([np.array([0]), fpr_test_static.mean(), np.array([1])])
tpr_test_static_y = np.concatenate([np.array([0]), tpr_test_static.mean(), np.array([1])])
auc_static_test = np.trapz(tpr_test_static_y, fpr_test_static_x)

ax[0].plot(fpr_train_static_x, tpr_train_static_y, marker='x', linewidth=2, label="static, auc: {:.2f}".format(auc_static_train))
ax[1].plot(fpr_test_static_x, tpr_test_static_y, marker='x', linewidth=2, label="static, auc: {:.2f}".format(auc_static_test))

# dynamic
fpr_train_dynamic = pd.read_csv('./output/dynamic/fpr_train.csv', header=None)
tpr_train_dynamic = pd.read_csv('./output/dynamic/tpr_train.csv', header=None)
fpr_train_dynamic_x = np.concatenate([np.array([0]), fpr_train_dynamic.mean(), np.array([1])])
tpr_train_dynamic_y = np.concatenate([np.array([0]), tpr_train_dynamic.mean(), np.array([1])])
auc_dynamic_train = np.trapz(tpr_train_dynamic_y, fpr_train_dynamic_x)

fpr_test_dynamic = pd.read_csv('./output/dynamic/fpr_test.csv', header=None)
tpr_test_dynamic = pd.read_csv('./output/dynamic/tpr_test.csv', header=None)
fpr_test_dynamic_x = np.concatenate([np.array([0]), fpr_test_dynamic.mean(), np.array([1])])
tpr_test_dynamic_y = np.concatenate([np.array([0]), tpr_test_dynamic.mean(), np.array([1])])
auc_dynamic_test = np.trapz(tpr_test_dynamic_y, fpr_test_dynamic_x)

ax[0].plot(fpr_train_dynamic_x, tpr_train_dynamic_y, marker='x', linewidth=2, label="dynamic, auc: {:.2f}".format(auc_dynamic_train))
ax[1].plot(fpr_test_dynamic_x, tpr_test_dynamic_y, marker='x', linewidth=2, label="dynamic, auc: {:.2f}".format(auc_dynamic_test))

# cvpr
#fpr_train_cvpr = pd.read_csv('./output/cvpr/fpr_train.csv', header=None)
#tpr_train_cvpr = pd.read_csv('./output/cvpr/tpr_train.csv', header=None)
#fpr_train_cvpr_x = np.concatenate([np.array([0]), fpr_train_cvpr.mean(), np.array([1])])
#tpr_train_cvpr_y = np.concatenate([np.array([0]), tpr_train_cvpr.mean(), np.array([1])])
#auc_cvpr_train = np.trapz(tpr_train_cvpr_y, fpr_train_cvpr_x)

#fpr_test_cvpr = pd.read_csv('./output/cvpr/fpr_test.csv', header=None)
#tpr_test_cvpr = pd.read_csv('./output/cvpr/tpr_test.csv', header=None)
#fpr_test_cvpr_x = np.concatenate([np.array([0]), fpr_test_cvpr.mean(), np.array([1])])
#tpr_test_cvpr_y = np.concatenate([np.array([0]), tpr_test_cvpr.mean(), np.array([1])])
#auc_cvpr_test = np.trapz(tpr_test_cvpr_y, fpr_test_cvpr_x)

#ax[0].plot(fpr_train_cvpr_x, tpr_train_cvpr_y, marker='x', linewidth=2, label="cvpr, auc: {:.2f}".format(auc_cvpr_train))
#ax[1].plot(fpr_test_cvpr_x, tpr_test_cvpr_y, marker='x', linewidth=2, label="cvpr, auc: {:.2f}".format(auc_cvpr_test))

# add legends
ax[0].legend(loc='lower right')
ax[0].grid()
ax[0].set_title('Train, mean of {} runs'.format(tpr_train_static.shape[0]))
ax[1].legend(loc='lower right')
ax[1].grid()
ax[1].set_title('Test, mean of {} runs'.format(tpr_test_static.shape[0]))

plt.show()