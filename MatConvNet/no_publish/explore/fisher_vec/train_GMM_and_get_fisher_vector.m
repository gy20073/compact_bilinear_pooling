function [trainFV, trainY, valFV, valY]=...
            train_GMM_and_get_fisher_vector(dataset, network, endLayer, nMixture, useRelu)
    % requires that the last conv activations is extracted
    [outputBase]=consts(dataset);
    
    % just for present, will support more in the future
    if ((~strcmp(dataset, 'MIT') && ~strcmp(dataset, 'CUB')) ...
            || ~strcmp(network, 'VGG-M'))
        error('only support MIT/CUB and VGG-M at present');
    end
    
    % file name of output fisher feature
    fisher_file=fullfile(outputBase, ['fisher_', dataset, '_',...
        network, '_', num2str(endLayer), '.mat']);
    
    if exist(fisher_file, 'file') == 2
        fprintf('loading from the existing mat file\n');
        load(fisher_file);
    else
        % feature output file name
        feature_in_file=fullfile(outputBase, ['activations_', dataset, '_',...
            network, '_', num2str(endLayer), '.mat']);
        % will load these 4 vars [trainFV, trainY, valFV, valY]
        fprintf('loading convolution outputs\n')
        load(feature_in_file);
        if useRelu
            trainFV=max(trainFV, 0); 
            valFV=max(valFV, 0);
        end
        trainFV=flatIt(trainFV);
        valFV=flatIt(valFV);
        
        % begin fit GMM
        fprintf('begin fitting GMM models')
        [means, covariances, priors] = ...
        vl_gmm(reshape(trainFV, size(trainFV, 1),[]), nMixture, ...
               'CovarianceBound', 0.0001, ...
               'NumRepetitions', 1,...
               'Verbose');
        gmm_file_name=fullfile(outputBase, ['gmm_', dataset, '_',...
            network, '_', num2str(endLayer), '.mat']);
        save(gmm_file_name, 'means','covariances','priors','-v7.3')
        % end of GMM
        
        % extract fisher features
        fprintf('extracting fisher vec for training data\n');
        trainFV=extract_fisher(trainFV, means, covariances, priors, nMixture);
        fprintf('extracting fisher vec for validation data\n');
        valFV=extract_fisher(valFV, means, covariances, priors, nMixture);
        
        save(fisher_file, 'trainFV', 'trainY', 'valFV', 'valY','-v7.3');
    end
end

function [outputBase]=consts(dataset)
    outputBase='data/exp_yang_fv';       
end

function trainFV=flatIt(trainFV)
    trainFV=permute(trainFV, [3,1,2,4]);
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
