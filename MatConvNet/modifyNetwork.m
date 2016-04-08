function net=modifyNetwork(network, dataset, varargin)
    % parse the input options
    mopts.poolType='bilinear'; % or fisher, compact_RM, compact_TS
    mopts.ftConvOnly=false; % if true, then FC layers won't be finetuned
    mopts.classifyType='LR'; % or SVM
    mopts.initMethod='pretrain'; % or 'random'
    mopts.use448=false;
    
    % only for compact_bilinear
    mopts.projDim=5000; 
    mopts.learnW=true;
    
    % only for pure fully connected network
    % i.e. use back prop to train a two layer fully connected NN
    mopts.isPretrainFCs=false;
    mopts.fc1Num=-1;
    mopts.fc2Num=-1;
    mopts.fcInputDim=-1;
    
    mopts.batchSize=128;
    mopts.nonLinearClassify='';%'some_path_to_pretrained_nonlinear_classify';
    mopts = vl_argparse(mopts, varargin);
    
    if mopts.isPretrainFCs
        % if we're training a two layer NN, then we could not use offline 
        % solver to initialize the weight of the last layer
        % Thus, change the initialization method to 'random'
        mopts.initMethod='random';
    end
    
    % load the network
    net=load(consts(dataset, network));
    num_classes=consts(dataset, 'num_classes');
    endLayer=consts(dataset, 'endLayer', 'network', network);
    lastNchannel=consts(dataset, 'lastNchannel', 'network', network);
    
    
    % throw away layers except 1:endLayer
    net.layers=net.layers(1:endLayer);    
   
    % modify the network to the required shape
    if mopts.isPretrainFCs
        % in this case, we will only train a two layer NN, without the
        % previous convolutional layers. So we throw away the loaded net. 
        net=struct('layers', {{}});
        if (mopts.fcInputDim<=0)
            error('input dimension not specified');
        end
        fprintf('Input dimension %d\n', mopts.fcInputDim);
        if (mopts.fc1Num>0)
            net=addFC(net, 'fc1', [mopts.fcInputDim, mopts.fc1Num], 'random');
            net=addReLU(net, 'relu_after_fc1');
            net=addDropout(net, 0.5);
            fprintf('FC_1 num neurons %d\n', mopts.fc1Num);
            if mopts.fc2Num>0
                net=addFC(net, 'fc2', [mopts.fc1Num, mopts.fc2Num], 'random');
                net=addReLU(net, 'relu_after_fc2');
                net=addDropout(net, 0.5);
                fprintf('FC_2 num neurons is %d\n', mopts.fc2Num);
                outDim=mopts.fc2Num;
            else
                outDim=mopts.fc1Num;
            end
        else
            outDim=mopts.fcInputDim;
        end
        % then the classification layer should also be initialized by
        % random weights
    elseif strcmp(mopts.poolType, 'fisher')
        net=addFisher(net, dataset, network, mopts.use448, mopts.batchSize);
        net=addL2norm(net);
        % TODO: currently only support Nchannel=512, fisher_compo=64
        outDim=65536;
    elseif strcmp(mopts.poolType, 'bilinear')
        net=addReLU(net,'relu_after_conv');
        net=addBilinear(net);
        net=addSqrt(net);
        net=addL2norm(net);
        outDim=lastNchannel^2;
    elseif strcmp(mopts.poolType, 'compact_bilinear') || ...
           strcmp(mopts.poolType, 'compact_RM')
        net=addReLU(net,'relu_after_conv');
        net=addCompactRM(net, mopts.projDim, lastNchannel, mopts.learnW, dataset);
        net=addSqrt(net);
        net=addL2norm(net);
        outDim=mopts.projDim;
    elseif strcmp(mopts.poolType, 'compact2') || ...
           strcmp(mopts.poolType, 'compact_TS')
        net=addReLU(net,'relu_after_conv');
        net=addCompactTS(net, mopts.projDim, lastNchannel, mopts.learnW, dataset);
        net=addSqrt(net);
        net=addL2norm(net);
        outDim=mopts.projDim;
    elseif strcmp(mopts.poolType, 'fcfc')
        % the FCFC baseline, only throw away the last classification layer
        net=load(consts(dataset, network));
        net.layers=net.layers(1:(endLayer+6));
        % could probably also add the sqrt & l2norm
        outDim=4096; % this assumes the underlying network is vgg-m or vgg-16
    elseif strcmp(mopts.poolType, 'fcfc_norm')
        net=load(consts(dataset, network));
        net.layers=net.layers(1:(endLayer+6));
        % try to add the sqrt & l2norm
        net=addSqrt(net);
        net=addL2norm(net);
        outDim=4096; % this assumes the underlying network is vgg-m or vgg-16
    elseif strcmp(mopts.poolType, 'pca_fb')
        net=addReLU(net,'relu_after_conv');
        net=addPCA(net, lastNchannel, mopts.projDim, mopts.use448,...
                   dataset, network, mopts.batchSize, mopts.learnW);
        
        net=addBilinear(net);
        net=addSqrt(net);
        net=addL2norm(net);
        
        outDim=mopts.projDim;
    else
        error('unknown pooling type');
    end
    
    if isempty(mopts.nonLinearClassify)
        % setup the last classification layer
        initFCparam=getFCinitWeight(mopts.initMethod, outDim, num_classes,...
            mopts.classifyType, network, mopts.poolType, dataset, net, mopts.use448, mopts.batchSize);
        net=addClassification(net, mopts.classifyType, initFCparam);
    else
        % load a trained nonlinear classifier from disk
        fprintf('using nonlinear classifier from %s\n', mopts.nonLinearClassify);
        t=load(mopts.nonLinearClassify);
        net.layers=[net.layers t.net.layers];
        clear t
    end
    
    % set whether learn the last fully classification layer
    if mopts.ftConvOnly
        assert( strcmp(net.layers{end-1}.name, 'fc_final') );
        % set learning rate of classification layer to 0
        net.layers{end-1}.learningRate=[0 0];
    end
    
    net=vl_simplenn_move(net, 'cpu');
end

%%%%%%%%%%%%%%%%%%%%%%% Various Layer Definition %%%%%%%%%%%%%%%%%%%%%%

function net=addPCA(net, lastNchannel, projDim, use448, dataset, network, batchSize, learnW)
    pcaOut=floor(sqrt(projDim));
    assert(pcaOut^2==projDim);
    % pca is a filter of size: 1*1*lastNchannel*sqrtProjDim
    % with a bias of length sqrtProjDim. All in single. 
        
    pcaWeightFile=consts(dataset, 'pcaWeight',...
        'network', network, ...
        'cfgId', 'pca_fb', ...
        'projDim', projDim, ...
        'use448', use448);
    
    if exist(pcaWeightFile, 'file') == 2
        fprintf('Load PCA weight from saved file: %s\n', pcaWeightFile);
        load(pcaWeightFile);
    else
        % get activations from the last conv layer % checkpoint
        [trainFV, trainY, valFV, valY]=...
            get_activations_dataset_network_layer(...
            dataset, network, 'pcaConv', use448, net, batchSize);
       
        samples=permute(trainFV, [1,2,4,3]); % from hwcn to hwnc
        samples=reshape(samples, [], size(samples, 4));
        ave=mean(samples, 1); % 1*dim vector
        coeff=pca(samples); % dim*dim matrix, each column is a principle direction
        bias=-ave*coeff; % a row vector, should be the initial value for bias
        
        coeff=single(coeff); 
        bias=single(bias);
        
        savefast(pcaWeightFile, 'coeff', 'bias');
        fprintf('save PCA weight to new file: %s\n', pcaWeightFile);
    end
    initPCAparam={{reshape(coeff(:, 1:pcaOut), 1, 1, lastNchannel, pcaOut),...
                   bias(1:pcaOut)}};
    
    net.layers{end+1} = struct('type', 'conv', 'name', 'pca_reduction', ...
       'weights', initPCAparam, ...
       'stride', 1, ...
       'pad', 0, ...
       'learningRate', [1 2]*learnW);
end

function net=addCompactTS(net, projDim, nc, learnW, dataset)
    TS_obj=yang_compact_bilinear_TS_nopool_2stream(projDim, [nc nc], 1);

    wf=consts(dataset, 'projInitW', 'projDim', projDim, 'cfgId', 'compact2');
    if exist(wf, 'file') ==2
        fprintf('loading an old compact2 weight\n');
        load(wf);
        % modify the initialized parameters to the loaded params 
        TS_obj.h_={hs(1,:), hs(2,:)};
        TS_obj.weights_={ss(1,:), ss(2,:)};
        TS_obj.setSparseM(TS_obj.weights_, 1);
    end
    
    net.layers{end+1}=struct('type', 'custom',...
        'layerObj', TS_obj, ...
        'forward', @TS_obj.forward_simplenn, ...
        'backward', @TS_obj.backward_simplenn, ...
        'name', 'compact_TS', ...
        'outDim', projDim, ...
        'weights', {{cat(1, TS_obj.weights_{:})}}, ...
        'learningRate', [1]*learnW);
end

function net=addCompactRM(net, projDim, nc, learnW, dataset)
    wf=consts(dataset, 'projInitW', 'projDim', projDim, 'cfgId', 'compact_bilinear');
    if exist(wf, 'file') ==2
        fprintf('loading an old compact bilinear weight\n');
        load(wf);
    else
        factor=1.0/sqrt(projDim);
        fprintf('generating a new compact bilinear weight\n');
        init_w={{factor*(randi(2,nc, projDim)*2-3),...
                 factor*(randi(2,nc, projDim)*2-3)}};
        savefast(wf, 'init_w');
    end
    
    net.layers{end+1}=struct('type', 'custom',...
        'forward', @yang_compact_bilinear_RM_forward, ...
        'backward', @yang_compact_bilinear_RM_backward, ...
        'name', 'compact_RM', ...
        'weights', init_w,...
        'outDim', projDim, ...
        'learnW', learnW, ...
        'learningRate', [1 1]*learnW);
end

function net=addBilinear(net)
    net.layers{end+1}=struct('type', 'custom',...
        'forward', @yang_bilinear_forward, ...
        'backward', @yang_bilinear_backward, ...
        'name', 'bilinear');
end

% initMethod: 'random' or 'pretrain'
% classificationType: SVM or LR
% network: 'VGG_M'
% cfgId: 'bilinear', 'compact_bilinear', 'fisher', 'fully_connected'
% dataset: 'MIT' 'CUB'
function initFCparam=getFCinitWeight(...
         initMethod,...
         nfeature, nclass,...
         classificationType, network, cfgId, dataset, netBeforeClassification,...
         use448, batchSize)
    
    if strcmp(initMethod, 'random')
        % random initialize
        initFCparam={{init_weight('xavierimproved', 1, 1, nfeature, nclass, 'single'),...
                    zeros(nclass, 1, 'single')}};
    elseif strcmp(initMethod, 'pretrain')
        weight_file=consts(dataset, 'classifierW',...
            'pretrainMethod', classificationType, ...
            'network', network, ...
            'cfgId', cfgId, ...
            'projDim', nfeature, ...
            'use448', use448);
        
        if exist(weight_file, 'file') == 2
            % svm or logistic initialized weight, load from disk
            load(weight_file);
        else
            % get activations from the last conv layer % checkpoint
            [trainFV, trainY, valFV, valY]=...
                get_activations_dataset_network_layer(...
                    dataset, network, cfgId, use448, netBeforeClassification, batchSize);
            % train SVM or LR weight, and test it on the validation set. 
            [w, b, acc, map, scores]= train_test_vlfeat(classificationType, ...
                squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
            % reshape the parameters to the input format
            w=reshape(single(w), 1, 1, size(w, 1), size(w, 2));
            b=single(squeeze(b));
            initFCparam={{w, b}};
            % save on disk
            savefast(weight_file, 'initFCparam', 'acc', 'map', 'scores');
        end
        % end of using pretrain weight
    else
        error('init method unknown');
    end
end

function net=addFisher(net, dataset, network, use448, batchSize)
    clear yang_fv
    % currently fix the number of mixtures=64, 
    init_param=train_GMM_and_get_fisher_vector_v2(...
               dataset, network, 64, use448, net, batchSize);
    
    net.layers{end+1}=struct('type','custom',...
        'init_param', {init_param},...
        'forward', @yang_fv_forward,...
        'backward',@yang_fv_backward,...
        'name', 'fisher_layer');
end

function net=addFC(net, name, initFCparam, initMethod)
    if strcmp(initMethod, 'random')
        initFCparam={{init_weight('xavierimproved', 1, 1, initFCparam(1), initFCparam(2), 'single'),...
                      zeros(initFCparam(2), 1, 'single')}};
    elseif strcmp(initMethod, 'specified')
        % nothing should be done
    else
        error('In addFC, unknown parameter initialization method.');
    end
    
    net.layers{end+1} = struct('type', 'conv', 'name', name, ...
       'weights', initFCparam, ...
       'stride', 1, ...
       'pad', 0, ...
       'learningRate', [1 2]);
end

% lossType could be: 'mhinge', 'softmaxlog', 'mshinge'
function net=addClassification(net, lossType, initFCparam)
    % convert the 'SVM' and 'LR' to internal representation
    if strcmp(lossType, 'LR')
        lossType='softmaxlog';
    elseif strcmp(lossType, 'SVM')
        lossType='mhinge';
    else
        error('unknown loss type');
    end

    net=addFC(net, 'fc_final', initFCparam, 'specified');
    
    % add the last softmax classification layer
    net.layers{end+1}=struct('type', 'loss',...
                             'name', 'final_loss', ...
                             'lossType', lossType);
end

function net=addDropout(net, rate)
    net.layers{end+1}=struct('type','dropout','rate', rate);
end

function net=addSqrt(net)
    net.layers{end+1}=struct('type', 'custom',...
        'forward',  @yang_sqrt_forward, ...
        'backward', @yang_sqrt_backward, ...
        'name', 'sign_sqrt');
end

function net=addL2norm(net)
    % implement my own layer, much faster
    net.layers{end+1}=struct('type', 'custom',...
        'forward',  @yang_l2norm_forward, ...
        'backward', @yang_l2norm_backward, ...
        'name', 'L2_normalize');
end

function net=addReLU(net, name)
    net.layers{end+1}=struct('type', 'relu', 'name', name);
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
    switch lower(opts)
      case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
      case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
      case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
      otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
    end
end
