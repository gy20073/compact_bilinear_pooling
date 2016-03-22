function train_test_gtsvm(trainFs, trainY, testFs, testY)
    C=1;
    epsilon=1E-3;
    
    tic;
    
    trainY=trainY-1; % the class starts from 0
    %Multiclass is only implemented for problems without an unregularized bias
    
    context = gtsvm;
    context.initialize( trainFs', trainY, true, C, 'polynomial', 1, 0, 1, false );
    context.optimize( epsilon, 10000 );
    classifications = context.classify( testFs');
    
    fprintf('using GTSVM: accuracy is %f, mAP is %f\n', ...
        score_accuracy(classifications, testY), ...
        mean(score_mAP(classifications, testY)));
    
    clear context;
    
    toc
end
