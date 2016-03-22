function train_test_libsvm_kernel(trainFV, trainY, valFV, valY, libpath)
    chi1=proj_chi2_mean(trainFV.^2, 10, 0.2, true);
    chi2=proj_chi2_mean(valFV.^2, 10, 0.2, true);
    %%
    [chi1_root, chi2_root]=normalize_root(chi1, chi2);
    [chi1_l2, chi2_l2]=normalize_l2(chi1_root, chi2_root);
    %chi1_l2=chi1; chi2_l2=chi2;
    %% create the chi2 mean pooling kernels
    ktrain=chi1_l2'*chi1_l2;
    ktest=chi2_l2'*chi1_l2;
    ktrain(1:10, 1:10)
    ktrain=[(1:size(ktrain, 1))' ktrain];
    ktest=[(1:size(ktest, 1))' ktest];
    %% create the exp-chi2 kernels
    gamma=-2;
    
    ktrain=exp(gamma*(1-chi1_l2'*chi1_l2));
    ktest=exp(gamma*(1-chi2_l2'*chi1_l2));
    ktrain(1:10, 1:10)
    % fix the id issue, required by libsvm
    ktrain=[(1:size(ktrain, 1))' ktrain];
    ktest=[(1:size(ktest, 1))' ktest];
    %% package path
    if nargin<=4  % no libpath set
        libpath='/home/yang/local/my_software/libsvm-3.20/matlab';
    end
    addpath(libpath);
    %% classify
    tic
    SVM_opt='-s 0 -c 1 -t 4 -m 1000 -e 0.001';
    
    trainedModel=ovrtrain(trainY, double(ktrain), SVM_opt);
    [predicted_label, accuracy, decision_values]= ...
        ovrpredict(valY, double(ktest), trainedModel);
    
    fprintf('using libsvm: accuracy is %f, mAP is %f\n', ...
        score_accuracy(decision_values, valY), ...
        mean(score_mAP(decision_values, valY)));
    toc
end


