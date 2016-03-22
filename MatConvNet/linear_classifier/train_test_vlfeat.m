function [wm, bm, acc, map, classifications]=train_test_vlfeat(methods, trainFs, trainY, testFs, testY)
% linear SVM classification wrapper of vlfeat's SVM
    % some hyper parameters
    reg_coeff=1/size(trainFs, 2);
    if strcmp(methods, 'LR')
        reg_coeff=reg_coeff/1000;
    end
    epsilon=1E-3;
    
    tic
    
    if strcmp(methods, 'SVM')
        loss_type='HINGE';
    elseif strcmp(methods, 'LR')
        loss_type='LOGISTIC';
    else
        error('unknown classification methods');
    end
    
    nclass=max(trainY);
    classifications=zeros(size(testFs, 2), nclass);
    
    ws=cell(1, nclass);
    bs=cell(1, nclass);

    % based on the size of training set, choose parfor and for
    varinfo=whos('trainFs');
    mem=varinfo.bytes/1024/1024/1024; % size in GB
    % fix the threshold to 4GB
    useSingle=(mem>4);
    fprintf('In vlfeat, train set=%f GB, use_single_thread=%d\n', mem, useSingle);
    
    if ~useSingle % using multiple threads
        parfor c=1:nclass
            fprintf('training class %d\n', c);
            y = 2*(trainY==c)-1;
             [ws{c},bs{c}] = vl_svmtrain(trainFs, y, reg_coeff, 'verbose',...
            'Loss', loss_type, 'Solver', 'SDCA', 'Epsilon', epsilon, ...
            'MaxNumIterations', 1E6);
        end
    else % using a single thread, code is the same, excpet the parfor -> for
        for c=1:nclass
            fprintf('training class %d\n', c);
            y = 2*(trainY==c)-1;
            
            [ws{c},bs{c}] = vl_svmtrain(trainFs, y, reg_coeff, 'verbose',...
            'Loss', loss_type, 'Solver', 'SDCA', 'Epsilon', epsilon, ...
            'MaxNumIterations', 1E6);
        end    
    end
    
    fprintf('training complete, making predictions\n')
    % not parfor to save memory
    for c=1:nclass
        y = 2*(trainY==c)-1;
        w=ws{c};
        b=bs{c};
        % try cheap calibration, don't affect the mAP
        predTrain=w'*trainFs+b;
        mp = median(predTrain(y>0)) ;
        mn = median(predTrain(y<0)) ;
        b = (b - mn) / (mp - mn) ;
        w = w / (mp - mn) ;
        ws{c}=w(:);
        bs{c}=b;
        % end of calibration
        pred = w'*testFs + b;
        
        classifications(:, c)=pred;
    end
    wm=cell2mat(ws);
    bm=cell2mat(bs);
    
    acc=score_accuracy(classifications, testY);
    map=mean(score_mAP(classifications, testY));
    fprintf('using vlfeat %s: accuracy=%f, mAP=%f\n', methods,  acc, map);
    
   toc
end