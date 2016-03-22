function train_test_libsvm_chi2(trainFs, trainY, testFs, testY, use1vsAll, libpath)
    tic
    SVM_opt='-s 0 -c 1 -t 7 -g 1 -e 0.001 -z 6';
    
    if nargin<6  % no libpath set
        libpath='/home/yang/local/my_software/libsvm_chi_ksirg/matlab';
    end
    addpath(libpath);
    
    if use1vsAll
        train=@ovrtrain;
        predict=@ovrpredict;
    else
        train=@svmtrain;
        predict=@svmpredict;
    end
    
    trainedModel=train(trainY, double(trainFs'), SVM_opt);
    [predicted_label, accuracy, decision_values]= ...
        predict(testY, double(testFs'), trainedModel);
        
    fprintf('using libsvm: accuracy is %f, mAP is %f\n', ...
        score_accuracy(decision_values, testY), ...
        mean(score_mAP(decision_values, testY)));
    toc
end


