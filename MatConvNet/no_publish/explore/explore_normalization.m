%%
setup_yang
dataset='MIT';
% Total: mit_indoor data: conv -> fisher_vector -> logistic regression
[trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer(dataset, 'VGG-M', 14);
% l2 normalization
[trainFV_l2, valFV_l2]=normalize_l2(trainFV, valFV);
% testing l2 norm
[svm_w, svm_b]=train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);
% the save is for initializing the parameters in the fine tuning process
save(consts(dataset, 'SVM_weights'),'svm_w','svm_b')

%% Testing sparsify features
[trainFV_sp, valFV_sp]=normalize_sparse(trainFV_l2, valFV_l2, 1e-3);
train_test_vlfeat('SVM', trainFV_sp, trainY, valFV_sp, valY);

%% Testing non square root version
[trainFV_sq, valFV_sq]=normalize_square(trainFV, valFV);
[trainFV_sq, valFV_sq]=normalize_l2(trainFV_sq, valFV_sq);
train_test_vlfeat('SVM', trainFV_sq, trainY, valFV_sq, valY);

%% Testing 1/4 root version
[trainFV_root, valFV_root]=normalize_root(trainFV, valFV);
[trainFV_root, valFV_root]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_root, trainY, valFV_root, valY);

%% check theoretical taylor correctness
[trainFV_sq, valFV_sq]=normalize_square(trainFV, valFV);
fprintf('original mean=%f, var=%f\n', mean(trainFV_sq(1,:)), var(trainFV_sq(1,:)));
[trainFV_taylor1, valFV_taylor1]=normalize_taylor_1st(trainFV_sq, valFV_sq);
fprintf('taylor mean=%f, var=%f\n', mean(trainFV_taylor1(1,:)), var(trainFV_taylor1(1,:)));
fprintf('true mean=%f, var=%f\n', mean(trainFV(1,:)), var(trainFV(1,:)));

%% check that square+root==identity
[trainFV_recon, valFV_recon]=normalize_root(trainFV_sq, valFV_sq);
fprintf('true mean=%f, var=%f\n', mean(trainFV_recon(1,:)), var(trainFV_recon(1,:)));

%% var normalization - linear approach, don't work (acc=61.5, mAP=61.6)
[trainFV_sq, valFV_sq]=normalize_square(trainFV, valFV);
[trainFV_vlinear, valFV_vlinear]=normalize_var1_linear(trainFV_sq, valFV_sq);

[trainFV_vlinear_l2, valFV_vlinear_l2]=normalize_l2(trainFV_vlinear, valFV_vlinear);
train_test_vlfeat('SVM', trainFV_vlinear_l2, trainY, valFV_vlinear_l2, valY);

%% component wise normalization, training never ends
[trainFV_compo, valFV_compo]=normalize_l2_component_wise(trainFV, valFV, 512);
[trainFV_compo, valFV_compo]=normalize_l2(trainFV_compo, valFV_compo);
train_test_vlfeat('SVM', trainFV_compo, trainY, valFV_compo, valY);

%% ranking approach (acc=12.8, mAP=60.2)
[trainFV_rank, valFV_rank]=normalize_rank(trainFV, valFV);
[trainFV_rank, valFV_rank]=normalize_l2(trainFV_rank, valFV_rank);
train_test_vlfeat('SVM', trainFV_rank, trainY, valFV_rank, valY);

%% normalize each dimension, not very good (acc=61.6, mAP=57.6)
[trainFV_div, valFV_div]=normalize_minus_div(trainFV, valFV, 1);
train_test_vlfeat('SVM', trainFV_div, trainY, valFV_div, valY);
