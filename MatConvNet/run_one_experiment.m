function run_one_experiment(dataset, network, gpuId, tag,...
            saveInter, batchSize, learningRate, weightDecay, varargin)
    prepare_dataset(dataset);
    %% few shot learning switch
    useFewShot=false;
    
    setup_yang;
    gpuDevice(gpuId);
    
    mopts.poolType='bilinear';
    % only for compact_bilinear
    mopts.projDim=8192;
    mopts.learnW=false;
    % only for pretrain multilayer perception
    mopts.isPretrainFCs=false;
    mopts.fc1Num=-1;
    mopts.fc2Num=-1;
    mopts.fcInputDim=512*512;

    % some usually fixed params
    mopts.ftConvOnly=false;
    mopts.use448=true;
    mopts.classifyType='LR'; % or SVM
    mopts.initMethod='pretrain'; % or 'random'
    mopts.nonLinearClassify='';
    
    mopts = vl_argparse(mopts, varargin);
    %% some parameters should be tuned
    opts.train.batchSize = batchSize;
    opts.train.learningRate = learningRate;
    opts.train.weightDecay = weightDecay;
    opts.train.momentum = 0.9 ; 
    
    % set the batchSize of initialization
    mopts.batchSize=opts.train.batchSize;

    % other parameters that usually is fixed
    opts.imdbPath = consts(dataset, 'imdb');
    opts.train.expDir = consts(dataset, 'expDirOne', 'cfgId', mopts.poolType, 'learnW', mopts.learnW,...
        'projDim', mopts.projDim, 'network', network, ...
        'pretrainMethod', mopts.classifyType, 'use448', mopts.use448, 'tag', tag);
    opts.train.numSubBatches = 1 ;
    opts.train.continue = true ;
    opts.train.gpus = gpuId ;  
    %gpuDevice(opts.train.gpus); % don't want clear the memory
    opts.train.prefetch = true ;
    opts.train.sync = false ; % for speed
    opts.train.cudnn = true ; % for speed
    opts.train.numEpochs = numel(opts.train.learningRate) ;
    opts.train.saveInter=saveInter;
    imdb = load(opts.imdbPath) ;
    % in case some dataset only has val/test
    opts.train.val=union(find(imdb.images.set==2), find(imdb.images.set==3));
    opts.train.train=[];
    
    % when using few shots
    valInter=1;
    if useFewShot, opts.train.train=few_train(imdb, 1); end

    % initialize network 
    net=modifyNetwork(network, dataset, mopts);
    
    % fine tune the fine tuned net
    % This section is legacy code, not used
    useFtFt=false;
    if useFtFt
        fine_tuned_net='/home/yang/exp_data/fv_layer/exp_yang_fv/CUB_VGG_M_LR_compact2_8192_fixW_448_final_1/net-epoch-60.mat';
        a=load(fine_tuned_net);
        net=a.net;
        multiLayer='/home/yang/exp_data/fv_layer/exp_yang_fv/CUB_VGG_M_LR_bilinear_448_vggm_ft_layer1_1024_drop/net-epoch-30.mat';
        net2=load(multiLayer);
        net2=net2.net;
        net.layers=[net.layers(1:(end-2)) net2.layers];
    end
    
    if strcmp(mopts.poolType, 'fisher')
        fprintf('fisher could not be fine tuned, return\n');
        return;
    end
    vl_simplenn_display(net);

    if mopts.use448 && ~mopts.isPretrainFCs
        net.normalization.averageImage=imresize(net.normalization.averageImage, ...
            448.0 / size(net.normalization.averageImage, 1));
        net.normalization.imageSize=[448 448 3];
    end

    if mopts.isPretrainFCs
        % if pretrain the multilayer perception, we should use a different
        % getBatch function, since the data is usually the activations of
        % some other network
        diskFile=consts(dataset, 'poolout', 'network', network,...
            'cfgId', mopts.poolType, 'use448', mopts.use448, 'projDim', mopts.projDim);
        fn=getBatchFromDisk(imdb, diskFile);
    else
        bopts=net.normalization;
        %bopts.transformation = 'stretch' ; %TODO (need such augmentation?)
        bopts.numThreads = 12;
        fn = getBatchWrapper(bopts) ;
    end
    
    %% learn the full model
    assert(strcmp(net.layers{end-1}.name, 'fc_final'));
    %net.layers{end-1}.learningRate=[0, 0];
    opts.train.backPropDepth=inf; % could limit the backprop
    [net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true, 'valInter', valInter);
end

function trainset=few_train(imdb, k)
    rng('default');
    nclass=max(imdb.images.label);
    
    trainset=[];
    for i=1:nclass
        cur=((imdb.images.set==1) & (imdb.images.label==i));
        cur=find(cur);
        trainset=[trainset cur(randsample(numel(cur), k))];
    end
end