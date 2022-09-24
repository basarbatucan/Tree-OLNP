import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from tree_partition.tree_partition import TreeNode, create_tree, add_indices

# delete later
import matplotlib.pyplot as plt

class tree_olnp_parallel:

    def __init__(self, tfpr_=0.1, eta_init_=0.01, beta_init_=100, sigmoid_h_=-1, Lambda_=0, tree_depth_=2, split_prob_=0.5, node_loss_constant_=1, projection_type_='PCA', ensemble_type_='probabilistic', max_row_= None, max_col_ = None) -> None:
        
        # hyperparameters
        self.tfpr = tfpr_
        self.eta_init = eta_init_
        self.beta_init = beta_init_
        self.sigmoid_h = sigmoid_h_
        self.Lambda = Lambda_
        self.tree_depth_ = tree_depth_
        self.split_prob_ = split_prob_
        self.node_loss_constant_ = node_loss_constant_
        self.projection_type_ = projection_type_
        self.ensemble_type_ = ensemble_type_

        # parameters
        self.w_ = None # perceptron weight
        self.b_ = None # perceptron bias
        self.connectivity_ = None
        self.partitioner_ = None
        self.P_ = None
        self.E_ = None
        self.partitions_ = None

        # below two parameters are used in space partitioning
        # algorithm needs to know size of the space
        self.max_row_ = max_row_
        self.max_col_ = max_col_
        
        # below arrays are used to store learning performance of npnn
        self.mu_train_array_ = None
        self.tpr_train_array_ = None
        self.fpr_train_array_ = None
        self.neg_class_weight_train_array_ = None # learned weight for negative class (note that we assume binary classes are 1 and -1)
        self.pos_class_weight_train_array_ = None # learned weight for positive class
        self.expert_weights_ = None

        # initial calculations
        self.number_of_nodes_ = 2**(tree_depth_+1)-1

    def fit(self, X, y, n_samples_augmented_min=150e3):

        # take the parameters
        tfpr = self.tfpr
        eta_init = self.eta_init
        beta_init = self.beta_init
        gamma = 1 # this is initialized as 1, can change later
        sigmoid_h = self.sigmoid_h
        Lambda = self.Lambda
        tree_depth = self.tree_depth_
        split_prob = self.split_prob_
        node_loss_constant = self.node_loss_constant_
        projection_type = self.projection_type_

        # data related parameters
        n_samples, n_features = X.shape

        # context tree, structural parameters
        number_of_nodes = 2**(tree_depth+1)-1

        # initiate perceptron parameters
        if projection_type == 'manual':
            # in manual projection, first 2D of the data is used as coords
            w = np.random.randn(n_features-2, number_of_nodes)*1e-4
        else:
            w = np.random.randn(n_features, number_of_nodes)*1e-4
        b = np.random.randn(number_of_nodes)*1e-4

        # prepare the X
        # note that since this is an online algorithm, to have better conversion, we augment initial data
        # initiate the Fourier features
        
        if n_samples<n_samples_augmented_min:
            # augmentation is necessary
            X_, y_ = self.__augment_data(X=X, y=y, n_samples=n_samples, n_features=n_features, n_samples_augmented_min=n_samples_augmented_min)
            n_samples = X_.shape[0] # get the new data size
        else:
            # augmentation is not necessary
            X_ = X
            y_ = y

        # get number of negative samples after augmentation (if necessary)
        number_of_negative_samples = np.sum(y_==-1)

        # initiate learning rates
        eta = eta_init
        beta_init = beta_init/number_of_negative_samples # scaling beta init with the number of total negative samples imrpoves convergence
        beta = beta_init
        
        # create negative sample buffer
        # this buffer is used to estimate FPR in an online manner during training, 200 is a constant and can be changed
        negative_sample_buffer_size = max(int(np.round(2/tfpr)), 200)
        negative_sample_buffer = np.zeros((negative_sample_buffer_size))
        negative_sample_buffer_index = 0

        # generate connectivity matrix which contains the connectivity information of the nodes within the context tree
        self.__generate_connectivity_matrix()
        connectivity = self.connectivity_

        # get all partitions
        root = create_tree(tree_depth)
        root = add_indices(root)
        partitions = root.partition()
        # get number of experts
        n_experts = len(partitions)
        # sort partitions
        partitions.sort(key=len)
        # save partitions
        self.partitions_ = partitions

        # calculate node centers
        # this information is used to determine how a sample propagates from top to root
        # Note that tree olnp requires an sample (subset of whole data set is ok) to initialize the tree
        # User can provide seperate feature set for context tree initialization
        self.__init_partitioner(X)

        P = np.ones((number_of_nodes,), dtype=np.float64)                                # node probability
        E = np.ones((number_of_nodes,), dtype=np.float64)                                # prediction performance of node n
        y_discriminant = np.zeros((number_of_nodes,))                  # discriminant of nodes
        C = np.ones((number_of_nodes,))                                # prediction of nodes
        node_loss_constant = node_loss_constant*(-320)/n_samples;      # adjustment to the learning rate to prevent numerical error
        sample_mu = np.zeros((n_samples, tree_depth+1))                # node weight used in ensemble classifier
        sigma_tree = np.zeros((tree_depth+1,))                            
        mu_tree = np.zeros((tree_depth+1,))
        expert_weights = np.zeros((n_samples, n_experts))

        # init accumulators
        tp = 0
        fp = 0
        tpr_train_array = np.zeros((n_samples))
        fpr_train_array = np.zeros((n_samples))
        neg_class_weight_train_array = np.zeros((n_samples))
        pos_class_weight_train_array = np.zeros((n_samples))
        gamma_array = np.zeros((n_samples))
        transient_number_of_positive_samples = 1
        transient_number_of_negative_samples = 1

        # initiate weights
        neg_class_weight_train_array[0] = 2*gamma
        pos_class_weight_train_array[0] = 2*gamma
        
        for i in range(0, n_samples):

            # take the input data
            xt = X_[i,:]
            yt = y_[i]

            # calculate sample propogation path through tree
            dark_node_indices = self.__find_dark_nodes(xt)

            # if manual projection is used, first 2D are coordinates
            if projection_type == 'manual':
                xt = xt[2:]

            # make prediction
            for k in range(0, len(dark_node_indices)):
                dark_node_index = dark_node_indices[k]
                # calculate sigma
                if k == 0:
                    sigma_tree[k] = 1-split_prob
                else:
                    sigma_tree[k] = (1-split_prob)*P[connectivity[dark_node_index, 1]]*sigma_tree[k-1]
                    if k == len(dark_node_indices)-1:
                        sigma_tree[k] = sigma_tree[k]/(1-split_prob)
                # calculate weights
                mu_tree[k] = sigma_tree[k]*E[dark_node_index]/P[0]
                if k == len(dark_node_indices)-1:
                    mu_tree[k] = 1-np.sum(mu_tree[:k])

                # calculate discriminant in each node
                y_discriminant_ = np.matmul(xt, w[:, dark_node_index])+b[dark_node_index]
                y_discriminant[dark_node_index] = y_discriminant_
                C[dark_node_index] = np.sign(y_discriminant_)
            
            # calculate prediction of the ensemble
            yt_predict = self.__expert_ensemble(C[dark_node_indices], y_discriminant[dark_node_indices], mu_tree)

            # save sample mu
            sample_mu[i, :] = mu_tree

            # save tp and fp
            if yt == 1:
                if yt_predict == 1:
                    tp = tp+1
            else:
                if yt_predict == 1:
                    fp = fp+1

            tpr_train_array[i] = tp/transient_number_of_positive_samples
            fpr_train_array[i] = fp/transient_number_of_negative_samples

            # save gamma
            gamma_array[i] = gamma

            # update the buffer with the current prediction
            if yt == -1:

                # modify the size of the FPR estimation buffer
                if negative_sample_buffer_index == negative_sample_buffer_size-1:
                    negative_sample_buffer[:-1] = negative_sample_buffer[1:]
                else:
                    negative_sample_buffer_index += 1

                if yt_predict == 1:
                    # false positive
                    negative_sample_buffer[negative_sample_buffer_index] = 1
                else:
                    # true negative
                    negative_sample_buffer[negative_sample_buffer_index] = 0

            # estimate the FPR of the current model using the moving buffer
            if negative_sample_buffer_index == negative_sample_buffer_size-1:
                estimated_FPR = np.mean(negative_sample_buffer)
            else:
                estimated_FPR = np.mean(negative_sample_buffer[:negative_sample_buffer_index+1])

            # y(t), calculate mu(t)
            if yt == 1:
                # mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
                mu = (transient_number_of_positive_samples + transient_number_of_negative_samples)/transient_number_of_positive_samples
                # save class costs
                pos_class_weight_train_array[i] = mu
                if i>1:
                    neg_class_weight_train_array[i] = neg_class_weight_train_array[i-1]
            else:
                # mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
                mu = gamma*(transient_number_of_positive_samples + transient_number_of_negative_samples)/transient_number_of_negative_samples
                # save class costs
                neg_class_weight_train_array[i] = mu
                if i>1:
                    pos_class_weight_train_array[i] = pos_class_weight_train_array[i-1]

            # SGD
            # Update perceptron in each node
            for k in range(len(dark_node_indices)-1,-1,-1):
                dark_node_index = dark_node_indices[k]
                # get the node loss
                z = yt*y_discriminant[dark_node_index]
                dloss_dz = self.__deriv_sigmoid_loss(z, sigmoid_h)
                node_loss = self.__sigmoid_loss(z, sigmoid_h)
                loss = mu*node_loss - gamma*tfpr
                # update node performance
                E[dark_node_index] = E[dark_node_index]*np.exp(node_loss_constant*loss)
                # update node prob
                if k == len(dark_node_indices)-1:
                    P[dark_node_index] = E[dark_node_index]
                else:
                    P[dark_node_index] = split_prob * P[connectivity[dark_node_index, 2]] * P[connectivity[dark_node_index, 3]] + (1-split_prob)*E[dark_node_index]
                # SGD on each node
                dz_dw = yt*xt
                dz_db = yt
                dloss_dw = dloss_dz*dz_dw
                dloss_db = dloss_dz*dz_db
                # update w and b
                w[:, dark_node_index] = (1-eta*Lambda)*w[:, dark_node_index]-eta*mu*dloss_dw
                b[dark_node_index] = b[dark_node_index]-eta*mu*dloss_db

            # y(t)
            if yt == 1:
                # calculate n_plus(t)
                transient_number_of_positive_samples += 1
            else:
                # calculate n_minus(t)
                transient_number_of_negative_samples += 1
                # calculate gamma(t)
                gamma = gamma*(1+beta*(estimated_FPR - tfpr))

            # update learning rate of perceptron
            eta = eta_init/(1+Lambda*(transient_number_of_positive_samples + transient_number_of_negative_samples))

            # update uzawa gain
            beta = beta_init/(1+Lambda*(transient_number_of_positive_samples + transient_number_of_negative_samples))

            # save performance of each expert
            for k in range(0, n_experts):
                partition_nodes = partitions[k]
                expert_weights[i, k] = np.log(np.prod(E[partition_nodes]))/(node_loss_constant*(i+1))

        # save updated parameters
        self.w_ = w
        self.b_ = b
        self.P_ = P
        self.E_ = E

        # save the results
        self.mu_train_array_ = sample_mu
        self.tpr_train_array_ = tpr_train_array
        self.fpr_train_array_ = fpr_train_array
        self.neg_class_weight_train_array_ = neg_class_weight_train_array
        self.pos_class_weight_train_array_ = pos_class_weight_train_array
        self.expert_weights_ = expert_weights

        return None

    def predict(self, X):
        # note that input should be numpy array, may need to fix this part for other data types
        n_test = X.shape[0]
        y_predict = np.empty(n_test)
        w = self.w_
        b = self.b_
        connectivity = self.connectivity_
        tfpr = self.tfpr
        P = self.P_
        E = self.E_
        projection_type = self.projection_type_

        number_of_nodes = 2**(self.tree_depth_+1)-1
        sigma_tree = np.zeros((self.tree_depth_+1,))
        mu_tree = np.zeros((self.tree_depth_+1, 1))
        y_discriminant = np.zeros((number_of_nodes, 1))
        C = np.zeros((number_of_nodes, 1))
        split_prob = self.split_prob_

        for i in range(0, n_test):
            # take the input data
            xt = X[i,:]

            # calculate sample propogation path through tree
            dark_node_indices = self.__find_dark_nodes(xt)

            # if manual projection is used, first 2D are coordinates
            if projection_type == 'manual':
                xt = xt[2:]

            # make prediction
            for k in range(0, len(dark_node_indices)):
                dark_node_index = dark_node_indices[k]
                # calculate sigma
                if k == 0:
                    sigma_tree[k] = 1-split_prob
                else:
                    sigma_tree[k] = (1-split_prob)*P[connectivity[dark_node_index, 1]]*sigma_tree[k-1]
                    if k == len(dark_node_indices)-1:
                        sigma_tree[k] = sigma_tree[k]/(1-split_prob)
                # calculate weights
                mu_tree[k] = sigma_tree[k]*E[dark_node_index]/P[0]
                if k == len(dark_node_indices)-1:
                    mu_tree[k] = 1-np.sum(mu_tree[:k])

                # calculate discriminant in each node
                y_discriminant_ = np.matmul(xt, w[:, dark_node_index])+b[dark_node_index]
                y_discriminant[dark_node_index] = y_discriminant_
                C[dark_node_index] = np.sign(y_discriminant_)
            
            # calculate prediction of the ensemble
            yt_predict = self.__expert_ensemble(C[dark_node_indices], y_discriminant[dark_node_indices], mu_tree)
            y_predict[i] = yt_predict

        return y_predict

    # aux functions
    def __expert_ensemble(self, expert_predictions, expert_discriminants, expert_weights):

        if self.ensemble_type_ == 'probabilistic':
            expert_weight_pmf = np.cumsum(expert_weights)
            uniform_rv = np.random.rand(1)[0]
            bool_array = (uniform_rv <= expert_weight_pmf)
            best_expert_index = np.where(bool_array == True)[0][0]
            ensemble_prediction = expert_predictions[best_expert_index]
            return ensemble_prediction

        elif self.ensemble_type_ == 'linear':
            ensemble_discriminant = np.dot(expert_discriminants, expert_weights)
            ensemble_prediction = np.sign(ensemble_discriminant)
            return ensemble_prediction

    def __find_dark_nodes(self, xt):
        dark_node_indices = np.zeros((self.tree_depth_+1,), dtype=np.int32)
        dark_node_index = 0
        current_node_index = 0

        # initialize with root node
        # every sample should visit the root node (simplest expert)
        dark_node_indices[dark_node_index] = current_node_index
        dark_node_index += 1

        if self.projection_type_ != 'manual':
            
            while self.connectivity_[current_node_index, 3] != -1:
                # project the node on the current pc
                xt_projected = np.matmul(xt, self.partitioner_['pc'][current_node_index])
                pc_mean = self.partitioner_['pc_mean'][current_node_index]
                # decide on node
                if xt_projected <= pc_mean:
                    # select the left node
                    next_node_index = self.connectivity_[current_node_index, 2]
                else:
                    # select the right node
                    next_node_index = self.connectivity_[current_node_index, 3]
                    
                # update the current node index
                current_node_index = next_node_index
                # save the node index
                dark_node_indices[dark_node_index,] = current_node_index
                # update dark node index
                dark_node_index +=1

        else:

            coords_xt = xt[0]
            coords_yt = xt[1]

            while self.connectivity_[current_node_index, 3] != -1:

                # calculate the distance to left child and right child
                left_child_index = self.connectivity_[current_node_index, 2]
                left_child_center_x = self.partitioner_['spatial_center'][left_child_index, 0]
                left_child_center_y = self.partitioner_['spatial_center'][left_child_index, 1]
                left_child_distance = np.sqrt((coords_xt-left_child_center_x)**2 + (coords_yt-left_child_center_y)**2)
                right_child_index = self.connectivity_[current_node_index, 3]
                right_child_center_x = self.partitioner_['spatial_center'][right_child_index, 0]
                right_child_center_y = self.partitioner_['spatial_center'][right_child_index, 1]
                right_child_distance = np.sqrt((coords_xt-right_child_center_x)**2 + (coords_yt-right_child_center_y)**2)

                # compare distances to both children
                if left_child_distance<=right_child_distance:
                    # closer to left child
                    next_node_index = left_child_index
                else:
                    # closer to right child
                    next_node_index = right_child_index
                
                # update the current node index
                current_node_index = next_node_index
                # save the node index
                dark_node_indices[dark_node_index,] = current_node_index
                # update dark node index
                dark_node_index +=1

        return dark_node_indices

    def __init_partitioner(self, X):
        # this function calculates the node centers based on first 2 principle components of X
        # X is projected to 2D space, and corresponding node centers are calculated
        # currently only PCA is used for projection
        max_depth = self.tree_depth_
        n_samples = X.shape[0]
        n_features = X.shape[1]

        if self.projection_type_ == 'PCA':

            # create array for saving dark indices
            dark_node_indices = np.ones((n_samples, max_depth+1))

            # project input to 2D space
            pca = PCA(n_components=2)
            pca.fit(X)

            # create node pc means
            partitioner = dict()
            partitioner['pc'] = np.zeros((self.number_of_nodes_, n_features))
            partitioner['pc_mean'] = np.zeros((self.number_of_nodes_,))
            node_index=0

            for i in range(0, max_depth):
                for j in range(1, 2**i+1):
                    
                    # select subset of x according to the partition
                    if i==0:
                        # root
                        current_set_index = np.ones(n_samples, dtype=bool)
                    else:
                        # divided data
                        current_set_index = dark_node_indices[:, i] == node_index

                    # calculate subset of X for lower parts of the tree
                    if sum(current_set_index) > 1:
                        # reset indices
                        left_child_index = np.zeros(n_samples, dtype=bool)
                        right_child_index = np.zeros(n_samples, dtype=bool)
                        # select active pc
                        pc = pca.components_[np.mod(i,2)] # alternate between principle components wrt tree depth
                        # get subset of X
                        subset_X = X[current_set_index, :]
                        # calculate projection of the subset
                        x_projected = np.matmul(subset_X, pc)
                        # calculate node mean
                        pc_mean = np.mean(x_projected)
                        # assign left and right subset from the current set (skip if leaf)
                        left_child_index[current_set_index] = x_projected <= pc_mean
                        right_child_index[current_set_index] = x_projected > pc_mean
                        dark_node_indices[left_child_index, i+1] = 2*node_index+1
                        dark_node_indices[right_child_index, i+1] = 2*node_index+2
                        # save the partitioner
                        partitioner['pc'][node_index] = pc
                        partitioner['pc_mean'][node_index] = pc_mean
                    else:
                        partitioner['pc'][node_index] = partitioner['pc'][node_index-1]
                        partitioner['pc_mean'][node_index] = partitioner['pc_mean'][node_index-1]

                    # update node index
                    node_index+=1

        elif self.projection_type_ == 'iterative_PCA':

            # create array for saving dark indices
            dark_node_indices = np.ones((n_samples, max_depth+1))

            partitioner = dict()
            partitioner['pc'] = np.zeros((self.number_of_nodes_, n_features))
            partitioner['pc_mean'] = np.zeros((self.number_of_nodes_,))
            node_index=0

            for i in range(0, max_depth):
                for j in range(1, 2**i+1):

                    # select subset of x according to the partition
                    if i==0:
                        # root
                        current_set_index = np.ones(n_samples, dtype=bool)
                    else:
                        # divided data
                        current_set_index = dark_node_indices[:, i] == node_index
                    if sum(current_set_index) > 1:
                        # reset indices
                        left_child_index = np.zeros(n_samples, dtype=bool)
                        right_child_index = np.zeros(n_samples, dtype=bool)
                        # calculate pca on the current node
                        pca = PCA(n_components=2).fit(X[current_set_index, :])
                        pc = pca.components_[0]
                        # project subset to the highes eigen vector with highest value
                        x_projected = np.matmul(X[current_set_index, :], pc)
                        # calculate node mean
                        pc_mean = np.mean(x_projected)
                        # assign left and right subset from the current set (skip if leaf)
                        left_child_index[current_set_index] = x_projected <= pc_mean
                        right_child_index[current_set_index] = x_projected > pc_mean
                        dark_node_indices[left_child_index, i+1] = 2*node_index+1
                        dark_node_indices[right_child_index, i+1] = 2*node_index+2
                        # save the partitioner
                        partitioner['pc'][node_index] = pc
                        partitioner['pc_mean'][node_index] = pc_mean
                    else:
                        partitioner['pc'][node_index] = partitioner['pc'][node_index-1]
                        partitioner['pc_mean'][node_index] = partitioner['pc_mean'][node_index-1]

                    # update node index
                    node_index+=1

        elif self.projection_type_ == 'manual':

            # create the space
            N = 100
            max_row = self.max_row_
            max_col = self.max_col_
            r = np.linspace(0, max_row, N)
            c = np.linspace(0, max_col, N)
            node_index=0
            [rr, cc] = np.meshgrid(r, c) # rows and cols contains all the points in space
            coords_rr = rr.flatten()
            coords_cc = cc.flatten()

            # create array for saving dark indices
            dark_node_indices = np.ones((N**2, max_depth+1))

            # in the manual projection type, user is providing 
            partitioner = dict()
            # this is 2D, because we are spatially partitioning the feature space
            partitioner['spatial_center'] = np.zeros((self.number_of_nodes_, 2))
            node_index=0

            for i in range(0, max_depth+1):
                selected_dim = np.mod(i,2)
                for j in range(1, 2**i+1):
                    # select subset of x according to the partition
                    if i==0:
                        # root
                        current_set_index = np.ones(N**2, dtype=bool)
                    else:
                        # divided data
                        current_set_index = dark_node_indices[:, i] == node_index

                    if sum(current_set_index) > 1:
                        # reset indices
                        left_child_index = np.zeros(N**2, dtype=bool)
                        right_child_index = np.zeros(N**2, dtype=bool)
                        # calculate the mean of the existing index
                        center_rr = coords_rr[current_set_index].mean()
                        center_cc = coords_cc[current_set_index].mean()
                        partitioner['spatial_center'][node_index, 0] = center_rr # note that we assume the first two dimension is center
                        partitioner['spatial_center'][node_index, 1] = center_cc
                        # assign left and right subset from the current set (skip if leaf)
                        if selected_dim == 0:
                            # assign with respect to rows
                            left_child_index[current_set_index] = coords_rr[current_set_index] <= center_rr
                            right_child_index[current_set_index] = coords_rr[current_set_index] > center_rr
                        else:
                            # assign with respect to cols
                            left_child_index[current_set_index] = coords_cc[current_set_index] <= center_cc
                            right_child_index[current_set_index] = coords_cc[current_set_index] > center_cc
                        if i<max_depth:
                            dark_node_indices[left_child_index, i+1] = 2*node_index+1
                            dark_node_indices[right_child_index, i+1] = 2*node_index+2
                    else:
                        partitioner['spatial_center'][node_index, 0] = max_row*0.5
                        partitioner['spatial_center'][node_index, 1] = max_col*0.5

                    # update node index
                    node_index+=1

        else:

            partitioner = None

        # update node centers
        self.partitioner_ = partitioner

        # functional test
        self.__test_init_partitioner(X)
        
        return None

    def __test_init_partitioner(self, X):

        leaf_nodes = []
        for i in range(0, X.shape[0]):
            xt = X[i,:]
            dark_node_indices = self.__find_dark_nodes(xt)
            leaf_node = dark_node_indices[-1]
            leaf_nodes.append(leaf_node)
        leaf_nodes = np.array(leaf_nodes)

        if self.projection_type_ != 'manual':
            # run region visualiztion for 2D cases only
            if X.shape[1] != 2:
                return None

            # for each X, identify the visited leaf node and visualize it
            f,ax = plt.subplots(1,1,figsize=(12,8))
            for leaf_index in set(leaf_nodes):
                target_index = leaf_nodes == leaf_index
                ax.scatter(X[target_index, 0], X[target_index, 1], marker='o', label = "{}".format(leaf_index))
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Node regions')
            plt.savefig('./figures/test__node_regions_visualized.png')
        else:
            leaf_dict = dict()
            leaf_dict['row'] = []
            leaf_dict['col'] = []
            leaf_dict['number_of_objects'] = []
            for leaf in leaf_nodes:
                leaf_dict['row'].append(self.partitioner_['spatial_center'][leaf,0])
                leaf_dict['col'].append(self.partitioner_['spatial_center'][leaf,1])
                leaf_dict['number_of_objects'].append(1)
            leaf_df = pd.DataFrame().from_dict(leaf_dict)
            leaf_df = leaf_df.groupby(['row', 'col']).agg({'number_of_objects':'sum'}).reset_index()
            leaf_df.plot(kind='scatter', x='col', y='row', s='number_of_objects')
            plt.xlim([0, self.max_col_])
            plt.xlabel('X')
            plt.ylim([0, self.max_row_])
            plt.ylabel('Y')
            plt.gca().invert_yaxis()
            plt.grid()
            plt.title('Location of leaf nodes (size is scaled with # of visited samples)')
            plt.savefig('./figures/test__leaf_nodes_visualized.png')

        return None

    def __generate_connectivity_matrix(self):
        number_of_nodes = self.number_of_nodes_
        total_number_of_node_relations = 4
        connectivity = np.ones((number_of_nodes, total_number_of_node_relations), dtype=np.int32)*-1
        node_index=0
        for i in range(0, self.tree_depth_+1):
            for j in range(1, 2**i+1):
                # define parent and sibling
                if np.mod(node_index, 2) == 1:
                    # left
                    parent = (node_index-1)/2
                    sibling = node_index+1
                else:
                    # right
                    parent = (node_index-2)/2
                    sibling = node_index-1
                    
                left_child = 2*node_index+1
                right_child = 2*node_index+2

                # handle root and lead nodes
                if parent < 0:
                    parent=-1
                    sibling=-1
                if right_child > number_of_nodes-1:
                    left_child=-1
                    right_child=-1

                connectivity[node_index, 0] = parent
                connectivity[node_index, 1] = sibling
                connectivity[node_index, 2] = left_child
                connectivity[node_index, 3] = right_child

                # update node index
                node_index+=1

        self.connectivity_ = connectivity.astype(int)

        return None

    def __augment_data(self, X, y, n_samples, n_features, n_samples_augmented_min):
        n_augmentation_call = int(n_samples_augmented_min//n_samples)+1
        n_samples_augmented = int(n_samples*n_augmentation_call)
        X_ = np.empty((n_samples_augmented, n_features))
        y_ = np.empty((n_samples_augmented))
        for i in range(0, n_augmentation_call):
            # shuffle index
            shuffle_index = np.arange(n_samples)
            np.random.shuffle(shuffle_index)
            # create augmented data
            X_[i*n_samples:(i+1)*n_samples, :] = X[shuffle_index, :]
            y_[i*n_samples:(i+1)*n_samples] = y[shuffle_index]
        return X_, y_

    def __deriv_sigmoid_loss(self, z, h):
        sigmoid_loss_x = self.__sigmoid_loss(z, h)
        return h*(1-sigmoid_loss_x)*sigmoid_loss_x

    def __sigmoid_loss(self, z, h):
        return 1/(1+np.exp(-h*z))