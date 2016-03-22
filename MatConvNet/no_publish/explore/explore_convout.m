% explore the case when output the raw convoutput
setup_yang
%%
% Total: mit_indoor data: conv -> fisher_vector -> logistic regression
% could optionally use the 448 image size
[trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer('MIT', 'VGG_M', 13);
[trainFV, valFV]=relu(trainFV, valFV);

%% test classify based on raw input
[trainFV_root, valFV_root]=normalize_root(trainFV, valFV);   
[trainFV_l2, valFV_l2]=normalize_l2(reshape(trainFV_root,[],size(trainFV_root, 4)),...
                                    reshape(valFV_root,[],size(valFV_root, 4)));
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% average pooling
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_ave, valFV_ave);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);

%% max pooling
[trainFV_max, valFV_max]=pool_max(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_max, valFV_max);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% ave pooling + outer product
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_outer, valFV_outer]=project_outer(trainFV_ave, valFV_ave);
[trainFV_root, valFV_root]=normalize_root(trainFV_outer, valFV_outer);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
%train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);
train_test_liblinear('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% bilinear pooling
[trainFV_bilinear, valFV_bilinear]=pool_bilinear(trainFV, valFV, true); 
%%
[trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);

%% bilinear adjacent version
[trainFV_bilinear, valFV_bilinear]=pool_bilinear_adjacent(trainFV, valFV); 
%%
[trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
% clear some variables, otherwise may have OOM issue
clear trainFV_bilinear valFV_bilinear trainFV_root valFV_root
clear trainFV valFV
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% bilinear + dense random projection
[trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
[trainFV_rp, valFV_rp]=pool_rand_project_dense(trainFV_root, valFV_root, 512);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_rp, valFV_rp);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% bilinear + sparsify
[trainFV_sp, valFV_sp]=normalize_sparse(trainFV_bilinear, valFV_bilinear, 1);
[trainFV_root, valFV_root]=normalize_root(trainFV_sp, valFV_sp);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% bilinear + select partial dimensions
ind=trainFV_l2>0;
ind2=sum(ind, 2);
select=ind2>1000;
count_big(select, 0)
trainFV_sub=trainFV_l2(select, :); valFV_sub=valFV_l2(select, :);
train_test_vlfeat('SVM', trainFV_sub, trainY, valFV_sub, valY);

%% ave pooling + bilinear dimension reduced
trainFV_comb=cat(1, trainFV_sub, trainFV_l2);
valFV_comb=cat(1, valFV_sub, valFV_l2);
train_test_vlfeat('SVM', trainFV_comb, trainY, valFV_comb, valY);

%% dimension reduction using PCA
% too time consuming, could only calculate top 512 singular values
A=sparse(double(cat(2, trainFV_sp, valFV_sp)));
[U,S,V]=svds(A,4096,'L',struct('tol', 1e-5));
t=S*V';
ind=rand(1,size(t, 2))>0.333;
[trainFV_root, valFV_root]=normalize_root(t(:, ind), t(:, ~ind));
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY(ind), valFV_l2, trainY(~ind));
%% bilinear + log
[trainFV_log, valFV_log]=normalize_matrix_log(trainFV_bilinear, valFV_bilinear, true, false);
[trainFV_root, valFV_root]=normalize_root(trainFV_log, valFV_log);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% bilinear dual pooling (pretty bad)
[trainFV_bilinear, valFV_bilinear]=pool_bilinear_dual(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% multi scale pooling
%  not as well as average pooling, because strong correlation
%  max pyramid is the same as average pooling
[trainFV_pyramid, valFV_pyramid]=pool_pyramid(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_pyramid, valFV_pyramid);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);
%% try pyramid pooling with compact bilinear feature
[trainFV_pyramid, valFV_pyramid]=pool_pyramid(trainFV, valFV);
%%
trainFV_pyramid2=trainFV_pyramid(1:8192*5, :); valFV_pyramid2=valFV_pyramid(1:8192*5,:);
[trainFV_root, valFV_root]=normalize_root(trainFV_pyramid2, valFV_pyramid2);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% orderless pooling methods
[trainFV_ol, valFV_ol]=pool_orderless(trainFV, valFV, 'x_x2');
[trainFV_root, valFV_root]=normalize_root(trainFV_ol, valFV_ol);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% random dimension reduction
[trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
[trainFV_dim_red, valFV_dim_red]=dim_reduction_random(trainFV_root, valFV_root, 20000);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_dim_red, valFV_dim_red);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% orderless pooling: root before pooling
%  not as well, or similar
[trainFV_root, valFV_root]=normalize_root(trainFV, valFV);
[trainFV_ol, valFV_ol]=pool_orderless(trainFV_root, valFV_root, 'x_x2');
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_ol, valFV_ol);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% average pooling with libsvm RBF kernel
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_ave, valFV_ave);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
% change the parameters in the following file first
train_test_libsvm(trainFV_l2, trainY, valFV_l2, valY, true);

%% try RBF kernel approximation
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_ave, valFV_ave);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);

[trainFV_appx, valFV_appx]=feature_RBF_approximation(trainFV_l2, valFV_l2, 10000);
[trainFV_appx, valFV_appx]=normalize_l2(trainFV_appx, valFV_appx);
train_test_vlfeat('SVM', trainFV_appx, trainY, valFV_appx, valY);

%% try RBF kernel approx + spatial pooling
[trainFV_rbf_spatial, valFV_rbf_spatial]=pool_orderless_RBF_approx(trainFV, valFV, 1000); 
train_test_vlfeat('SVM', trainFV_rbf_spatial, trainY, valFV_rbf_spatial, valY);

%% try deterministic bilinear dimension reduction
[trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
[trainFV_deter, valFV_deter]=dim_reduction_sum(trainFV_root, valFV_root, 20);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_deter, valFV_deter);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% try chi square on ave features
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
%[trainFV_root, valFV_root]=normalize_root(trainFV_ave, valFV_ave);
[trainFV_l1, valFV_l1]=normalize_l1(trainFV_ave, valFV_ave);
%trainFV_l1=10*trainFV_l1; valFV_l1=10*valFV_l1;
train_test_libsvm_chi2(trainFV_l1, trainY, valFV_l1, valY, true);
%% validate approximate chi2 on ave features
%  ave pooling features
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_l1, valFV_l1]=normalize_l1(trainFV_ave, valFV_ave);
%  project to form chi2 features
trainFV_chi2=proj_chi2(trainFV_l1, 5, 0.3);
valFV_chi2=proj_chi2(valFV_l1, 5, 0.3);
train_test_vlfeat('SVM', trainFV_chi2, trainY, valFV_chi2, valY);

%% orderless chi2
%[trainFV_l1conv, valFV_l1conv]=normalize_l1_conv(trainFV, valFV);
%[trainFV_l1conv, valFV_l1conv]=normalize_l1_indi(trainFV.^4, valFV.^4);
[trainFV_chi2, valFV_chi2]=pool_orderless(trainFV.^4, valFV.^4, 'chi2');

[trainFV_root, valFV_root]=normalize_root(trainFV_chi2, valFV_chi2);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% orderless exp-chi2 
%[trainFV_l1conv, valFV_l1conv]=normalize_l1_indi(trainFV, valFV);
[trainFV_l1conv, valFV_l1conv]=normalize_l1_conv(trainFV, valFV);

[trainFV_exp_chi2, valFV_exp_chi2]=pool_exp_chi2(trainFV_l1conv, valFV_l1conv, 10000);
train_test_vlfeat('SVM', trainFV_exp_chi2, trainY, valFV_exp_chi2, valY);

%% exp-chi2 approximation, validate on ave features
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_l1, valFV_l1]=normalize_l1(trainFV_ave, valFV_ave);
to4dim=@(x)reshape(x, 1,1, size(x, 1), size(x,2));
trainFV_l1=to4dim(trainFV_l1);
valFV_l1=to4dim(valFV_l1);
[trainFV_exp_chi2, valFV_exp_chi2]=pool_exp_chi2(trainFV_l1, valFV_l1, 30000);
train_test_vlfeat('SVM', trainFV_exp_chi2, trainY, valFV_exp_chi2, valY);

%% approximate the bilinear kernel
%  use gpu for poly2 leads to worse perform
[trainFV_poly, valFV_poly]=feature_poly2_approximation(trainFV, valFV, 5000);

[trainFV_root, valFV_root]=normalize_root(trainFV_poly, valFV_poly);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

