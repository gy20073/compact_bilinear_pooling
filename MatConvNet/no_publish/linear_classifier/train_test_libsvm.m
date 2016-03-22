function train_test_libsvm(trainFs, trainY, testFs, testY, use1vsAll, libpath)
    tic
    SVM_opt='-s 0 -c 1 -t 1 -d 4 -g 1 -r 1 -m 1000 -e 0.001'; % polynomial
    SVM_opt='-s 0 -c 1 -t 0 -m 1000 -e 0.001'; % linear kernel
    SVM_opt='-s 0 -c 1 -t 2 -g 1 -m 1000 -e 0.001'; % rbf, change -g -c
    
    if nargin<6  % no libpath set
        libpath='/home/yang/local/my_software/libsvm-3.20/matlab';
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


