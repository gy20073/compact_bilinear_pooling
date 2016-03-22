%%
setup_yang
% Total: mit_indoor data: conv -> fisher_vector -> logistic regression
[trainFV, trainY, valFV, valY]=train_GMM_and_get_fisher_vector( ...
    'CUB', 'VGG-M',13, 64, false);
train_test_vlfeat('SVM', trainFV, trainY, valFV, valY);

%% Testing sparse
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
