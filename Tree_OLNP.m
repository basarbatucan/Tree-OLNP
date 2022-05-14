classdef Tree_OLNP
    
    properties
        
        % NP classification parameters
        tfpr_
        n_features_
        
        % parameters
        w_
        b_
        connectivity_
        node_pc_means_
        P_
        E_
        
        % hyperparameters
        eta_init_
        beta_init_
        gamma_
        sigmoid_h_
        lambda_
        tree_depth_
        split_prob_
        node_loss_constant_
        
        % results
        mu_train_array_
        tpr_train_array_
        fpr_train_array_
        tpr_test_array_
        fpr_test_array_
        neg_class_weight_train_array_
        pos_class_weight_train_array_
        test_indices_
        
    end
    
    methods
        
        function obj = Tree_OLNP(eta_init, beta_init, gamma, sigmoid_h, lambda, tree_depth, split_prob, node_loss_constant, n_features, tfpr)
            
            % init hyperparameters
            obj.eta_init_ = eta_init;
            obj.beta_init_ = beta_init;
            obj.gamma_ = gamma;
            obj.sigmoid_h_ = sigmoid_h;
            obj.lambda_ = lambda;
            obj.tree_depth_ = tree_depth;
            obj.split_prob_ = split_prob;
            obj.node_loss_constant_ = node_loss_constant;
            
            % init model parameters
            obj.n_features_ = n_features;
            obj.tfpr_ = tfpr;
            
        end
        
        function obj = train(obj, X_train, y_train, X_test, y_test, test_repeat)
            
            % init NP classification parameters
            tfpr = obj.tfpr_;
            n_features = obj.n_features_;
            n_samples = size(X_train, 1);
            
            % init hyperparameters from constructor
            eta_init = obj.eta_init_;
            beta_init = obj.beta_init_;
            gamma = obj.gamma_;
            sigmoid_h = obj.sigmoid_h_;
            lambda = obj.lambda_;
            tree_depth = obj.tree_depth_;
            split_prob = obj.split_prob_;
            node_loss_constant = obj.node_loss_constant_;
            
            % init training parameters
            n_samples_train = size(X_train, 1);
            n_samples_test = size(X_test, 1);
            number_of_negative_samples = sum(y_train==-1);
            tree_node_number = 2^(tree_depth+1)-1;
            
            % context tree, structural parameters
            connectivity = generate_tree(obj);                             % connectivity matrix explaning the connections between regions
            node_pc_means = obj.init_tree_pc_means(X_train);               % calculate node specific principle component for space separation
            P = ones(tree_node_number, 1);                                 % node probability
            E = ones(tree_node_number, 1);                                 % prediction performance of node n
            y_discriminant = zeros(tree_depth+1,1);                        % discriminant of nodes
            C = ones(tree_node_number,1);                                  % prediction of nodes
            node_loss_constant = node_loss_constant*(-320)/n_samples;      % adjustment to the learning rate to prevent numerical error
            sample_mu = zeros(n_samples, tree_depth+1);                    % node weight used in ensemble classifier
            sigma_tree = zeros(tree_depth+1,1);                            
            mu_tree = zeros(tree_depth+1,1);
            
            % perceptron parameters
            w = randn(n_features, tree_node_number)*1e-4;                  % weight of each node
            b = randn(1, tree_node_number)*1e-4;                           % bias of each node
            
            eta = eta_init;
            beta_init = beta_init/number_of_negative_samples;              % after augmentation, we scale learning rate for class specific weight
            beta = beta_init;
            negative_sample_buffer_size = max(round(2/tfpr), 200);
            negative_sample_buffer = zeros(1, negative_sample_buffer_size);
            negative_sample_buffer_index = 1;
            
            % save initial model parameters
            obj.w_ = w;
            obj.b_ = b;
            obj.connectivity_ = connectivity;
            obj.node_pc_means_ = node_pc_means;
            obj.P_ = P;
            obj.E_ = E;
            
            % save test related parameters for online evaluation
            index_act_pos = y_test == 1;
            N_act_pos = sum(index_act_pos);
            index_act_neg = y_test == -1;
            N_act_neg = sum(index_act_neg);

            % init accumulators
            tp = 0;
            fp = 0;
            test_i = logspace(1, log10(n_samples_train), test_repeat+1);
            test_i = round(test_i(2:end));
            current_test_i = 1;

            % array outputs
            tpr_test_array = zeros(1, test_repeat);
            fpr_test_array = zeros(1, test_repeat);
            tpr_train_array = zeros(1, n_samples_train);
            fpr_train_array = zeros(1, n_samples_train);
            neg_class_weight_train_array = zeros(1, n_samples_train);
            pos_class_weight_train_array = zeros(1, n_samples_train);
            gamma_array = zeros(1, n_samples_train);
            number_of_positive_samples = 1;
            number_of_negative_samples = 1;

            %add initials
            neg_class_weight_train_array(1) = 2*gamma;
            pos_class_weight_train_array(1) = 2*gamma;

            % online training
            for i=1:n_samples_train

                % take the input data
                xt = X_train(i, :);
                yt = y_train(i, :);
                
                % find dark nodes
                dark_node_indices = obj.find_dark_nodes(xt);
                
                % make the prediction
                for k=1:length(dark_node_indices)

                    dark_node_index = dark_node_indices(k);

                    % calculate sigma
                    if k==1
                        sigma_tree(k) = 1-split_prob;
                    else
                        sigma_tree(k) = (1-split_prob)*P(connectivity(dark_node_index, 2))*sigma_tree(k-1);
                        if k==length(dark_node_indices)
                            sigma_tree(k)=sigma_tree(k)/(1-split_prob);
                        end
                    end

                    % calculate weights
                    mu_tree(k) = sigma_tree(k)*E(dark_node_index)/P(1);
                    if k==length(dark_node_indices)
                        mu_tree(k) = 1-sum(mu_tree(1:k-1));
                    end
                    
                    % calculate discriminant in each node
                    y_discriminant_ = xt*w(:,dark_node_index)+b(dark_node_index);
                    y_discriminant(dark_node_index) = y_discriminant_;
                    C(dark_node_index) = sign(y_discriminant_);
                    
                end
                
                % probabilistic ensemble
                yt_predict_index = [];
                while isempty(yt_predict_index)
                    yt_predict_index = dark_node_indices(find(rand<cumsum(mu_tree),1,'first'));
                    if isempty(yt_predict_index)
                        % this section is triggered if context tree
                        % framework fails to converge
                        fprintf('%f,', mu_tree);
                        fprintf('\n');
                        fprintf('%f,', sigma_tree);
                        fprintf('\n');
                        error('Tree convergence failed...\nbeta_init: %d\ngamma: %.1f\nsigmoid_h: %1.f\nlambda: %3.f\ntree depth: %d\nsplit prob: %.1f\nnode loss constant: %.2f', ...
                            obj.beta_init_,  obj.gamma_, obj.sigmoid_h_, obj.lambda_, ...
                            obj.tree_depth_, obj.split_prob_, obj.node_loss_constant_);
                    end
                end
                yt_predict = C(yt_predict_index);
                
                % save sample mu
                sample_mu(i, :) = mu_tree;
                
                % save tp and fp
                if yt == 1
                    if yt_predict == 1
                        tp = tp+1;
                    end
                else
                    if yt_predict == 1
                        fp = fp+1;
                    end
                end
                tpr_train_array(i) = tp/number_of_positive_samples;
                fpr_train_array(i) = fp/number_of_negative_samples;

                % save gamma
                gamma_array(i) = gamma;

                % test case
                if i==test_i(current_test_i)

                    % run the prediction for test case
                    y_predict_tmp = zeros(n_samples_test,1);
                    sigma_tree__ = zeros(tree_depth+1,1);
                    mu_tree__ = zeros(tree_depth+1,1);
                    y_discriminant__ = zeros(tree_depth+1,1);
                    C__ = zeros(tree_depth+1,1);
                    for j=1:n_samples_test
                        % get the sample
                        xt_tmp = X_test(j,:);
                        % find dark nodes
                        dark_node_indices__ = obj.find_dark_nodes(xt_tmp);
                        for k=1:length(dark_node_indices__)
                            dark_node_index = dark_node_indices__(k);
                            % calculate sigma
                            if k==1
                                sigma_tree__(k) = 1-split_prob;
                            else
                                sigma_tree__(k) = (1-split_prob)*P(connectivity(dark_node_index, 2))*sigma_tree__(k-1);
                                if k==length(dark_node_indices__)
                                    sigma_tree__(k)=sigma_tree__(k)/(1-split_prob);
                                end
                            end
                            % calculate weights
                            mu_tree__(k) = sigma_tree__(k)*E(dark_node_index)/P(1);
                            if k==length(dark_node_indices__)
                                mu_tree__(k) = 1-sum(mu_tree__(1:k-1));
                            end
                            % calculate discriminant in each node
                            y_discriminant__single = xt_tmp*w(:,dark_node_index)+b(dark_node_index);
                            y_discriminant__(dark_node_index) = y_discriminant__single;
                            C__(dark_node_index) = sign(y_discriminant__single);
                        end
                        % probabilistic ensemble
                        yt_predict_index = dark_node_indices__(find(rand<cumsum(mu_tree__),1,'first'));
                        y_predict_tmp(j) = C__(yt_predict_index);
                    end

                    % evaluate
                    index_pred_pos = y_predict_tmp == 1;

                    tp_tmp = sum(index_pred_pos & index_act_pos);
                    tpr_tmp = tp_tmp/N_act_pos;
                    tpr_test_array(current_test_i) = tpr_tmp;

                    fp_tmp = sum(index_pred_pos & index_act_neg);
                    fpr_tmp = fp_tmp/N_act_neg;
                    fpr_test_array(current_test_i) = fpr_tmp;

                    current_test_i=current_test_i+1;

                end

                % update the buffer with the current prediction
                if yt == -1

                    % modify the size of the FPR estimation buffer
                    if negative_sample_buffer_index == negative_sample_buffer_size
                        negative_sample_buffer(1:end-1) = negative_sample_buffer(2:end);
                    else
                        negative_sample_buffer_index = negative_sample_buffer_index + 1;
                    end

                    if yt_predict == 1
                        % false positive
                        negative_sample_buffer(negative_sample_buffer_index) = 1;
                    else
                        % true negative
                        negative_sample_buffer(negative_sample_buffer_index) = 0;
                    end

                end

                % estimate the FPR of the current model using the moving buffer
                if negative_sample_buffer_index == negative_sample_buffer_size
                    estimated_FPR = mean(negative_sample_buffer);
                else
                    estimated_FPR = mean(negative_sample_buffer(1:negative_sample_buffer_index));
                end
                
                % y(t), calculate mu(t)
                if yt==1
                    % mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
                    mu = (number_of_positive_samples + number_of_negative_samples)/number_of_positive_samples;
                    % save class costs
                    pos_class_weight_train_array(i) = mu;
                    if i>1
                        neg_class_weight_train_array(i) = neg_class_weight_train_array(i-1);
                    end
                else
                    % mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
                    mu = gamma*(number_of_positive_samples + number_of_negative_samples)/number_of_negative_samples;
                    % save class costs
                    neg_class_weight_train_array(i) = mu;
                    if i>1
                        pos_class_weight_train_array(i) = pos_class_weight_train_array(i-1);
                    end
                end

                % SGD
                % Update perceptron in each node
                for k=length(dark_node_indices):-1:1
                    dark_node_index = dark_node_indices(k);
                    % get the node loss
                    z = yt*y_discriminant(dark_node_index);
                    dloss_dz = utility_functions.deriv_sigmoid_loss(z,sigmoid_h);
                    node_loss = utility_functions.sigmoid_loss(z,sigmoid_h);
                    loss = mu*node_loss - gamma*tfpr;
                    % update node performance
                    E(dark_node_index) = E(dark_node_index)*exp(node_loss_constant*loss);
                    % update node prob
                    if k==length(dark_node_indices)
                        P(dark_node_index) = E(dark_node_index);
                    else
                        P(dark_node_index) = split_prob * P(connectivity(dark_node_index, 3)) * P(connectivity(dark_node_index, 4)) + (1-split_prob)*E(dark_node_index);
                    end
                    % sgd on each node
                    dz_dw = yt*xt';
                    dz_db = yt;
                    dloss_dw = dloss_dz*dz_dw;
                    dloss_db = dloss_dz*dz_db;
                    % update w and b
                    w(:, dark_node_index) = (1-eta*lambda)*w(:, dark_node_index)-eta*mu*dloss_dw;
                    b(dark_node_index) = b(dark_node_index)-eta*mu*dloss_db;

                end

                % update learning rate of perceptron
                %eta = eta_init/(1+lambda*(number_of_positive_samples + number_of_negative_samples));

                % y(t)
                if yt==1
                    % calculate n_plus(t)
                    number_of_positive_samples = number_of_positive_samples + 1;
                else
                    % calculate n_minus(t)
                    number_of_negative_samples = number_of_negative_samples + 1;
                    % calculate gamma(t)
                    gamma = gamma*(1+beta*(estimated_FPR - tfpr));
                end

                % update uzawa gain
                beta = beta_init/(1+lambda*(number_of_positive_samples + number_of_negative_samples));

            end
            
            % save calculated parameters
            obj.w_ = w;
            obj.b_ = b;
            obj.connectivity_ = connectivity;
            obj.node_pc_means_ = node_pc_means;
            obj.P_ = P;
            obj.E_ = E;
            
            % save the results
            obj.mu_train_array_ = sample_mu;
            obj.tpr_train_array_ = tpr_train_array;
            obj.fpr_train_array_ = fpr_train_array;
            obj.tpr_test_array_ = tpr_test_array;
            obj.fpr_test_array_ = fpr_test_array;
            obj.neg_class_weight_train_array_ = neg_class_weight_train_array;
            obj.pos_class_weight_train_array_ = pos_class_weight_train_array;
            obj.test_indices_ = test_i;
            
        end
        
        % generate tree structure
        function connectivity = generate_tree(obj)
            row = 2^(obj.tree_depth_+1)-1;
            col = 4;
            connectivity = ones(row, col)*-1;
            node_index=0;
            for i=0:obj.tree_depth_
                for j=1:2^i
                    % define the row index
                    node_index=node_index+1;
                    % define parent and sibling
                    if mod(node_index,2)==0
                        %left
                        parent = node_index/2;
                        sibling = node_index+1;
                    else
                        %right
                        parent = (node_index-1)/2;
                        sibling = node_index-1;
                    end
                    left_child = node_index*2;
                    right_child = node_index*2+1;
                    % handle root and lead nodes
                    if parent<1
                        parent=-1;
                        sibling=-1;
                    end
                    if right_child>row
                        left_child=-1;
                        right_child=-1;
                    end
                    connectivity(node_index, 1) = parent;
                    connectivity(node_index, 2) = sibling;
                    connectivity(node_index, 3) = left_child;
                    connectivity(node_index, 4) = right_child; 
                end
            end
        end
        
        % calculate iterative PCA partitions
        function node_pc_means = init_tree_pc_means(obj, x)
            % take parameters
            max_depth = obj.tree_depth_;
            n_samples = size(x,1);
            % calculate node means
            node_pc_means = cell(1, 2^(max_depth+1)-1);
            dark_node_indices = ones(n_samples, max_depth+1);
            node_index=0;
            for i=0:max_depth-1
                for j=1:2^i
                    node_index=node_index+1;
                    % select subset of x according to the partition
                    if i==0
                        % root
                        current_set_index = true(n_samples, 1);
                    else
                        % divided data
                        current_set_index = dark_node_indices(:,i+1) == node_index;
                    end
                    if sum(current_set_index)>1
                        % reset indices
                        left_child_index = false(n_samples, 1);
                        right_child_index = false(n_samples, 1);
                        % calculate pca on the current node
                        [eig_vectors, ~, eig_values] = pca(x(current_set_index, :));
                        % select the eigen vector for the highest eigen value
                        [~, selected_eig_vector_i] = max(eig_values);
                        % get the pc
                        pc = eig_vectors(:,selected_eig_vector_i);
                        % project subset to the highes eigen vector with highest value
                        x_projected = x(current_set_index, :)*pc;
                        % calculate node mean
                        pc_mean = mean(x_projected);
                        % assign left and right subset from the current set (skip if leaf)
                        left_child_index(current_set_index) = x_projected <= pc_mean;
                        right_child_index(current_set_index) = x_projected > pc_mean;
                        dark_node_indices(left_child_index, i+2) = node_index*2;
                        dark_node_indices(right_child_index, i+2) = node_index*2+1;
                        % save the partitioner
                        node_pc_means{node_index}.pc = pc;
                        node_pc_means{node_index}.pc_mean = pc_mean;
                    else
                        node_pc_means{node_index}.pc = node_pc_means{node_index-1}.pc+randn*1e-3;
                        node_pc_means{node_index}.pc_mean = node_pc_means{node_index-1}.pc_mean+randn*1e-3;
                    end
                end
            end
        end
        
        % find the nodes that incoming xt will visit
        function dark_node_indices = find_dark_nodes(obj, xt)
            dark_node_indices = zeros(1, obj.tree_depth_+1);
            dark_node_index = 1;
            current_node_index = 1;
            dark_node_indices(dark_node_index) = current_node_index;
            dark_node_index = dark_node_index+1;
            while ~(obj.connectivity_(current_node_index,4)==-1)
                % project the node on the current pc
                xt_projected = xt*obj.node_pc_means_{current_node_index}.pc;
                pc_mean = obj.node_pc_means_{current_node_index}.pc_mean;
                % decide on node
                if xt_projected <= pc_mean
                    % select the left node
                    next_node_index = obj.connectivity_(current_node_index, 3);
                else
                    % select the right node
                    next_node_index = obj.connectivity_(current_node_index, 4);
                end
                % update the current node index
                current_node_index = next_node_index;
                % save the node index
                dark_node_indices(dark_node_index) = current_node_index;
                dark_node_index = dark_node_index+1;
            end
        end
        
        function plot_results(obj)
            
            subplot(2,3,1)
            plot(obj.tpr_train_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Training Samples');
            ylabel('Train TPR');

            subplot(2,3,2)
            plot(obj.fpr_train_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Training Samples');
            ylabel('Train FPR');

            subplot(2,3,3)
            plot(obj.neg_class_weight_train_array_, 'LineWidth', 2);grid on;hold on;
            plot(obj.pos_class_weight_train_array_, 'LineWidth', 2);
            xlabel('Number of Training Samples');
            ylabel('Class weights');
            legend({'Neg class weight', 'Pos class weight'});
            
            subplot(2,3,4)
            plot(obj.tpr_test_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Tests');
            ylabel('Test TPR');

            subplot(2,3,5)
            plot(obj.fpr_test_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Tests');
            ylabel('Test FPR');
            
            figure()
            plot(obj.mu_train_array_, '*');
            legends = cell(1, obj.tree_depth_+1);
            for i=1:obj.tree_depth_+1
                legends{i} = sprintf('Depth %d',i-1);
            end
            legend(legends);
            
        end
        
    end
    
end