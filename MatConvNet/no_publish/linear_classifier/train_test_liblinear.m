function train_test_liblinear(method, trainFs, trainY, testFs, testY, libpath)
    tic
    SVM_opt='-s 1 -c 1 -B 1 -e 0.001';
    LR_opt='-s 7 -c 1 -B 1 -e 0.001';
    if strcmp(method, 'SVM')
        opt=SVM_opt;
    elseif strcmp(method, 'LR')
        opt=LR_opt;
    else
        error('In train_test_liblinear: unknown classification methods\n');
    end
    
    if nargin<6  % no libpath set
        libpath='~/local/my_software/liblinear-2.01/matlab/';
    end
    addpath(libpath);

    trainedModel=train(trainY, sparse(double(trainFs)), opt, 'col');
    [predicted_label, accuracy, decision_values]= ...
        predict(testY, sparse(double(testFs)), trainedModel, '','col');
        
    fprintf('using liblinear %s: accuracy is %f, mAP is %f\n', ...
        method,  score_accuracy(decision_values, testY), ...
        mean(score_mAP(decision_values, testY)));
    toc
end
