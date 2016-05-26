%% default configuration
setup_yang
mopts=struct('poolType','compact_TS', ... % could specify different pooling types
             ...% available options: bilinear, compact_RM, compact_TS, fisher, fcfc
             'use448', true, ... % input image resize to 448 (true) or 224 (false)
             ... % only for compact_RM & compact_TS
             'projDim', 8192,... % the projection dimension for compact features
             'learnW', true, ... % whether to learn the random weights in the compact layer
             ... % only for pretrain multi-layer percepturns
             'isPretrainFCs', false,... % true to train multilayer perception
             'fc1Num', -1,... % the number of neurons in the 1st hidden layer
             'fc2Num', -1,... % the number of neurons in the 2nd hidden layer
             'fcInputDim', 512*512,... % input feature dimension
             ... % after training the above multilayer perception, use this option
             ... % to specify the trained model location, this will concats to the 
             ... % network that has already been set up. 
             'nonLinearClassify', '', ... 
             ... % some options that usually don't change
             'ftConvOnly', false,... % if true, set lr of the classification layer to 0. 
             'classifyType', 'LR',...% 'SVM', 'LR': the loss used in the classification layer
             'initMethod', 'pretrain'); % 'pretrain', 'random': when set to 
                % pretrain, extract activation first and call vl_svmtrain
                % to initialize the parameters of the final classification
                % layer. When set to random, use Xavier initialization. 

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='CUB';
% network: 'VGG_M', 'VGG_16'
network='VGG_M';
% in the case of multiple gpus on a system, specify which one to use. 
gpuId=1;
% tag to diffrentiate multiple versions of experiments. 
tag='some_tag';
% Save the trained model every saveIter epoches. Mainly for saving disk space.
saveInter=10;

% batch size, usually VGG_M=32; VGG_16=8;
batchSize=32;
% set a reasonable learning rate
learningRate=[ones(1, 200)*1E-3];
% weightDecay: usually use the default value
weightDecay=0.0005;
%% change the options based on default values
gpuId=1;
tag='final_1';
batchSize=32;
mopts.poolType='fcfc';
mopts.use448=false;
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% change the options based on default values
gpuId=2;
tag='sample';
batchSize=8;
dataset='MIT';
mopts.poolType='bilinear';
mopts.use448=false;
network='VGG_16';
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts); 
%%           
tag='sample';
mopts.use448=false;
batchSize=32;

gpuId=1;
mopts.poolType='compact_RM';
dataset='DTD';
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% 
tag='sample';
mopts.use448=false;
batchSize=32;

gpuId=1;
mopts.poolType='fisher';
dataset='DTD';
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% 
tag='sample';
mopts.use448=false;
batchSize=32;

gpuId=1;
mopts.poolType='fcfc';
dataset='FMD';
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% test the "data doesn't exist on GPU problem"
tag='data_dont_exist';
mopts.use448=true;
batchSize=8;
network='VGG_16';

gpuId=6;
mopts.poolType='compact_bilinear';
dataset='MIT';
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts); 
%% rerun all experiments with 224 resolution
% all possible exp configurations
poolTypes = {'bilinear', 'fisher'};
datasets = {'CUB', 'MIT', 'DTD'};
networks = {'VGG_M', 'VGG_16'};
% specify the task for this machine
datasets = {'CUB'};
networks = {'VGG_M'};
gpuId = 3;
tag = 'final_1';
poolTypes = {'fcfc'};
mopts.learnW = false;

for ipool = 1 : numel(poolTypes)
    for idataset = 1 : numel(datasets)
        for inetwork = 1 : numel(networks)
            mopts.poolType = poolTypes{ipool};
            mopts.use448 = false;
            dataset = datasets{idataset};
            network = networks{inetwork};

            if strcmp(network, 'VGG_M')
                batchSize = 128;
                learningRate = ones(1, 100) * 1e-3;
            elseif strcmp(network, 'VGG_16')
                batchSize = 32;
                learningRate = ones(1, 30) * 1e-3;
            else
                error('unknown network type');
            end

            run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                               batchSize, learningRate, weightDecay, mopts);
        end
    end
end               