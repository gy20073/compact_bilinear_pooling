function [init_param]=...
            train_GMM_and_get_fisher_vector_v2(dataset, network, nMixture, use448, net, batchSize)
    fisherInit=consts(dataset, 'fisherInit', 'network', network, 'use448', use448);
    
    if exist(fisherInit, 'file')==2
        fprintf('loading an old fisher init param from file %s\n', fisherInit);
        load(fisherInit, 'means', 'covariances', 'priors');
    else
        fprintf('training a new fisher param file\n');
        % require net
        [trainFV, trainY, valFV, valY]=...
        get_activations_dataset_network_layer(dataset, network, 'fisherConv', use448, net, batchSize);
        
        % from hwcn to c*hw*n
        trainFV=flatIt(trainFV);
        valFV=flatIt(valFV);
        
        % subsample the training set
        ntrain=size(trainFV, 2)*size(trainFV, 3);
        subInd=randsample(ntrain, min(500000, ntrain));
        trainSub=reshape(trainFV, size(trainFV, 1),[]);
        trainSub=trainSub(:, subInd);
        
        % begin fit GMM
        fprintf('begin fitting GMM models')
        [means, covariances, priors] = ...
        vl_gmm(trainSub, nMixture, ...
               'CovarianceBound', 0.0001, ...
               'NumRepetitions', 1,...
               'Verbose');
        savefast(fisherInit, 'means','covariances','priors')
        
        poolout=consts(dataset, 'poolout', 'network', network, 'cfgId', 'fisher', ...
                       'use448', use448);
        if ~(exist(poolout, 'file')==2)
            % extract fisher features
            fprintf('extracting fisher vec for training data\n');
            trainFV=extract_fisher(trainFV, means, covariances, priors, nMixture);
            fprintf('extracting fisher vec for validation data\n');
            valFV=extract_fisher(valFV, means, covariances, priors, nMixture);

            savefast(poolout, 'trainFV', 'trainY', 'valFV', 'valY');
        end        
    end
    init_param={means, covariances, priors};
end

function trainFV=flatIt(trainFV)
    trainFV=permute(trainFV, [3,1,2,4]); % from hwcn to chwn to c*hw*n
    trainFV=reshape(trainFV, size(trainFV,1), [], size(trainFV, 4));
end

function Y=extract_fisher(fs, means, covariances, priors, nMixture)
    xN=size(fs, 3);
    Y=zeros(2*nMixture*size(fs,1), xN);
    for i=1:xN
        Y(:, i) = vl_fisher(fs(:,:,i), ...
                            means, ...
                            covariances, ...
                            priors, ...
                            'Improved');
    end
end
