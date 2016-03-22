%% default configuration
setup_yang
mopts=struct('poolType','bilinear', ... % bilinear compact2 compact_bilinear fisher fcfc
             'use448', true, ...
             ... % only for compact2 & compact_bilinear
             'projDim', 8192,... 
             'learnW', true, ...
             ... % only for pretrain multiple percepturns
             'isPretrainFCs', false,...
             'fc1Num', -1,...
             'fc2Num', -1,...
             'fcInputDim', 512*512,...
             'nonLinearClassify', '', ...
             ... % some general but usually fixed options
             'ftConvOnly', false,...
             'classifyType', 'LR',...
             'initMethod', 'pretrain');

% dataset: 'CUB', 'MIT', todo: 'FMD', 'ImageNet'
dataset='CUB';
% network: 'VGG_M', todo: 'VGG_16'
network='VGG_M';
% gpuId: 1,3,4,6 
gpuId=1;
% tag: depend on experiments
tag='some_tag';
% saveInter: 1
saveInter=1;

% batch size: use448 VGG_M=32; LR=256
batchSize=32;
% adjust feasible learning rates
%learningRate=ones(1, 60)*1E-3;
learningRate=[ones(1, 100)*1E-3];
% weightDecay: usually use default; when training the last layer LR,
% possile to set to 0
weightDecay=0.0005;
%% For Marcel2, fine tuning on the first 100 classes of CUB
mopts.poolType='compact_RM';
gpuId=1;
tag='Marcel2';

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                           batchSize, learningRate, weightDecay, mopts);
% remember to delete all outputs of this run, otherwise would confuse people                       
%% Lisa test set prediction labels
mopts.poolType='compact2';
network='VGG_16';
gpuId=3;
mopts.learnW=false;
tag='final_1';

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                           batchSize, learningRate, weightDecay, mopts);
%% code to save output variable
ofname='./data/exp_yang_fv/CUB_VGG_16_LR_compact2_8192_fixW_448_final_1/predictions.mat';
preds=lisa_out(1,:);
imdb=load(consts('CUB','imdb'));
image_names=imdb.images.name(imdb.images.set==3);
ground_truth=imdb.images.label(imdb.images.set==3);
save(ofname, 'preds', 'image_names', 'ground_truth', 'imdb');

%% rebuttal continue to run TS 4096 and 128^2
mopts.poolType='compact2';
tag='final_1';
mopts.learnW=false;

dims=[128^2];
gpuId=7;

for dim=dims
    mopts.projDim=dim;
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                           batchSize, learningRate, weightDecay, mopts);
end
%% rebuttal run pca_fb
mopts.poolType='pca_fb';
tag='rebuttal_pca_fb';
mopts.learnW=false;

dims=256;
gpuId=2;

for dim=dims
    mopts.projDim=dim;
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                           batchSize, learningRate, weightDecay, mopts);
end
%% rebuttal PCA-FB baseline
% TS 256, 1024
% SHOULD RUN INITIALIZATION (the 1st section) BEFORE TESTING
mopts.poolType='compact2';
tag='rebuttal_TS_baseline';
mopts.learnW=false;

mopts.projDim=1024;
gpuId=4;

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);

%% fisher matching
mopts.poolType='compact2';
mopts.use448=true;
dataset='MIT';
network='VGG_M';
gpuId=3;
tag='MATCH_FISHER';
batchSize=1;
mopts.classifyType='SVM';

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);

%% MIT_full
mopts.poolType='fcfc';
mopts.use448=false;
dataset='MIT_FULL';
network='places_16';
gpuId=5;
tag='finetuneDivergeAna';
batchSize=32;
learningRate=ones(1, 60)*1E-3;

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% fcfc with sqrt and l2norm
mopts.poolType='fcfc_norm';
mopts.use448=false;
dataset='MIT_FULL';
network='places_16';
gpuId=7;
tag='finetuneDivergeAna';
batchSize=32;

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);

%% fine tune on MIT with places
mopts.poolType='compact2';
mopts.use448=false;
dataset='MIT';
network='places_16';
gpuId=6;
tag='finetuneDivergeAna';
batchSize=32;
learningRate=ones(1, 60)*1E-3;

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% fine tune on CUB head
mopts.poolType='compact2';
mopts.use448=true;
mopts.learnW=true;
dataset='CUB_HEAD';
%dataset='CUB_BODY';
network='VGG_16';
gpuId=5;
tag='ning';
batchSize=16;
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);


%% run some custom experiments
% mit*{all encoders}*vgg_m /224/8192/noLearnW/batch32
dataset='MIT';
network='VGG_16';
gpuId=7;

learningRate=ones(1, 60)*1E-3;
batchSize=16;
poolTypes={'compact_bilinear'};
%poolTypes={'bilinear','compact2','compact_bilinear'};
% temp
%poolTypes={'compact2', 'compact_bilinear', 'fisher', 'fcfc'};

for i=1:numel(poolTypes)
    saveInter=1;
    mopts.poolType=poolTypes{i};
    mopts.learnW=false;
    tag='final_1_batch16';
    
    mopts.use448=true;
    if strcmp(mopts.poolType, 'fcfc')
        mopts.use448=false;
    end
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
end
%% test
% mit*{all encoders}*vgg_m /224/8192/noLearnW/batch32
dataset='MIT';
network='VGG_M';
gpuId=4;

learningRate=ones(1, 60)*1E-3;
poolTypes={'compact2'};
tag='test_convOnly_b64';
batchSize=64;
mopts.ftConvOnly=true;
% temp
%poolTypes={'compact2', 'compact_bilinear', 'fisher', 'fcfc'};

for i=1:numel(poolTypes)
    saveInter=2;
    mopts.poolType=poolTypes{i};
    mopts.learnW=false;
    
    mopts.use448=true;
    if strcmp(mopts.poolType, 'fcfc')
        mopts.use448=false;
    end
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
end
%% test the MIT fine tuning problem
mopts.poolType='compact_bilinear';
mopts.projDim=8192;
mopts.learnW=false;
gpuId=5;
tag='test';
dataset='MIT';
network='VGG_M';
batchSize=32;
mopts.use448=false;
learningRate=[ones(1, 60)*1E-3];
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);

%% run some custom experiments
% CUB*{all encoders}*vgg_m /224/8192/noLearnW/batch32
learningRate=[ones(1, 100)*1E-3];
poolTypes={'fisher'};
for i=1:numel(poolTypes)
    mopts.poolType=poolTypes{i};
    mopts.learnW=false;
    gpuId=1;
    tag='test_fisher';
    mopts.classifyType='LR';
    network='VGG_M';

    dataset='CUB';
    mopts.use448=true;
    if strcmp(mopts.poolType, 'fcfc')
        mopts.use448=false;
    end
    
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
end
%% run some custom experiments
% CUB*{all encoders}*vgg_16 /224/8192/noLearnW/batch32
learningRate=ones(1,30)*1E-3;
poolTypes={'fcfc'};
for i=1:numel(poolTypes)
    mopts.poolType=poolTypes{i};
    mopts.learnW=false;
    gpuId=6;
    batchSize=32;
    tag='final_1';

    dataset='CUB';
    network='VGG_16';
    mopts.use448=true;
    if strcmp(mopts.poolType, 'fcfc')
        mopts.use448=false;
    end
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
end
%% test the compact2 with learnW
mopts.poolType='compact2';
mopts.learnW=true;
gpuId=5;
batchSize=32;
tag='final_1';

dataset='CUB';
network='VGG_M';
mopts.use448=true;
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% test FCFC on CUB VGG-M
saveInter=3;

mopts.poolType='fcfc';
gpuId=3;
batchSize=32;
tag='final_1';

dataset='CUB';
network='VGG_M';
mopts.use448=false;
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% compact2: fixed W
%dims=2.^[7 9 11 12 13 14];
%dims=2.^[14 13 12];
%dims=[4096 2048 512];
%dims=[512];
dims=[2048];

learningRate=ones(1, 100)*1E-3;
gpuId=8;
saveInter=3;
for dim=dims
    mopts.projDim=dim;
    mopts.poolType='compact2';
    mopts.learnW=false;
    batchSize=32;
    tag='final_1';

    dataset='CUB';
    network='VGG_M';
    mopts.use448=true;
    mopts
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                           batchSize, learningRate, weightDecay, mopts);
end 
%% compact1
%dims=2.^[7 9 11];
%dims=2.^[14 13 12];
dims=[2048 4096];

gpuId=6;
saveInter=3;
for dim=dims
    mopts.projDim=dim;
    mopts.poolType='compact_bilinear';
    mopts.learnW=false;
    batchSize=32;
    tag='final_1';

    dataset='CUB';
    network='VGG_M';
    mopts.use448=true;
    mopts
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                           batchSize, learningRate, weightDecay, mopts);
end
%% pick up small projected dimension
methods={'compact2'};
for im=1:numel(methods)
    method=methods{im};
    
    %dims=2.^[7];
    dims=2.^[9];
    
    gpuId=5;
    saveInter=3;
    for dim=dims
        mopts.projDim=dim;
        mopts.poolType=method;
        mopts.learnW=true;
        batchSize=32;
        tag='final_1';

        dataset='CUB';
        network='VGG_M';
        mopts.use448=true;
        method
        mopts
        run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                               batchSize, learningRate, weightDecay, mopts);
    end
end

%% FMD * allEncoders * VGG_M
learningRate=[ones(1, 60)*1E-5];
saveInter=5;
poolTypes={'bilinear', 'compact_bilinear', 'fcfc'};
for i=1:numel(poolTypes)
    mopts.poolType=poolTypes{i};
    mopts.learnW=false;
    gpuId=5;
    tag='final_1';

    dataset='FMD';
    network='VGG_M';                   
    batchSize=32;
    mopts.use448=false;

    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
end
%% DTD * all * VGGM
poolTypes={'compact_bilinear', 'fcfc'};
mopts.use448=false;
mopts.learnW=false;
dataset='DTD';
gpuId= 8;
tag='final_1';
saveInter=2;

for i=1:numel(poolTypes)
    mopts.poolType=poolTypes{i};
    run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
end
%% missing woft
mopts.poolType='bilinear';
mopts.learnW=false;
gpuId=1;
tag='final_1';
dataset='CUB';
network='VGG_M';
batchSize=32;
mopts.use448=true;
mopts.classifyType='LR';
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% testing instead using SVM
mopts.poolType='compact2';
mopts.learnW=false;
gpuId=1;
tag='test';
dataset='CUB';
network='VGG_M';
batchSize=32;
mopts.use448=true;
mopts.classifyType='LR';
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% train logistic regression
mopts.poolType='bilinear';
mopts.use448=true;
mopts.isPretrainFCs=true;
mopts.fc1Num=-1;
mopts.fc2Num=-1;
mopts.fcInputDim=512*512;

dataset='CUB';
network='VGG_M';
gpuId=2;
tag='bp_logistic';
saveInter=10;
batchSize=256;
learningRate=ones(1,500);
weightDecay=0;
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% logistic my configurations
mopts.poolType='bilinear';
mopts.use448=true;
mopts.isPretrainFCs=true;
mopts.fc1Num=-1;
mopts.fc2Num=-1;
mopts.fcInputDim=512*512;

dataset='CUB';
network='VGG_M';
gpuId=2;
tag='bp_logistic_myconf';
saveInter=10;
batchSize=32;
learningRate=ones(1,500);
weightDecay=0;
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% multi layer network using compact2
mopts.poolType='compact2';
mopts.use448=true;
mopts.projDim=8192;
mopts.isPretrainFCs=true;
dataset='CUB';
network='VGG_M';
gpuId=2;
saveInter=30;
mopts.fcInputDim=mopts.projDim;

tag='layer2_4k_1k_drop_nodrop';  
mopts.fc1Num=4096;
mopts.fc2Num=1024;

batchSize=256;
weightDecay=1/6000/1000;
% generate the weights
%learningRate=[ones(1,90)*1E0 ones(1, 200)*1E-1];
iters=600;
t0=2/weightDecay;
learningRate=1./(weightDecay*((1:iters)+t0));
%% fine tuning using nonlinear classifier
mopts.poolType='compact2';
mopts.use448=true;
mopts.projDim=8192;
mopts.learnW=false;
mopts.isPretrainFCs=false;
mopts.nonLinearClassify=['/home/yang/exp_data/fv_layer/exp_yang_fv/'...
...%'CUB_VGG_M_LR_compact2_8192_fixW_448_layer1_4096/net-epoch-90.mat'];
'CUB_VGG_M_LR_compact2_8192_learnW_448_multi_layer_1_dropout_2048/net-epoch-60.mat']

learningRate=[ones(1, 60)*1E-3 ones(1, 50)*1E-4];
%learningRate=ones(1,100)*1E-3;
dataset='CUB';
network='VGG_M';
gpuId=3;
tag='ft_nonlinear_2048_90';
saveInter=2;

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% extract fine tuned activation, then train whatever classifier
% load the fine tuned network
%fpath='data/exp_yang_fv/CUB_VGG_16_LR_compact2_8192_fixW_448_final_1/net-epoch-11.mat';
fpath='data/exp_yang_fv/CUB_VGG_M_LR_compact2_8192_fixW_448_final_1/net-epoch-60.mat';
a=load(fpath);
a.net.layers=a.net.layers(1:(end-2));

% extract activations
dataset='CUB';
network='VGG_M';
use448=true;
netBeforeClassification=a.net;
batchSize=256;
gpuId=2;
gpuDevice(gpuId);

cfgId='compact2_ft';
[trainFV, trainY, valFV, valY]=...
                get_activations_dataset_network_layer(...
                    dataset, network, cfgId, use448, netBeforeClassification, batchSize);
%% try simple svm classification
train_test_vlfeat('LR', trainFV, trainY, valFV, valY);
%% try using original weights
b=load(fpath);
b=b.net.layers{end-1};
w=squeeze(b.weights{1});
b0=b.weights{2};
cls=@(trainFV, w, b0)bsxfun(@plus, trainFV'*w, b0);
score_accuracy(cls(trainFV, w, b0), trainY)
score_accuracy(cls(valFV, w, b0), valY)

%% try multilayer network
% set the diskFile var in run_one_experiment
mopts.isPretrainFCs=true;
gpuId=1;
saveInter=30;
mopts.fcInputDim=8192;

tag='vggm_ft_layer1_512_drop';  
mopts.fc1Num=512;
mopts.fc2Num=-1;

batchSize=256;
weightDecay=1/6000/1000;
% generate the weights
%learningRate=[ones(1,90)*1E0 ones(1, 200)*1E-1];
iters=600;
t0=2/weightDecay;
learningRate=1./(weightDecay*((1:iters)+t0));
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% try fine tuned previous network + (on the fine tuned) trained 4096dropout classifynet
% set the diskFile var in run_one_experiment
mopts.isPretrainFCs=false;
gpuId=4;
saveInter=3;

tag='finetune_the_finetune'; 

batchSize=32;
weightDecay=1/6000/1000;
% generate the weights
%learningRate=[ones(1,90)*1E0 ones(1, 200)*1E-1];
iters=600;
t0=2/weightDecay;
learningRate=1./(weightDecay*((1:iters)+t0));

learningRate=ones(1,100)*1E-3;
run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% few shots fine tuning
% first enable it in the run_one_experiment
% and enable it in get_activations_dataset_network_layer
% then hide the possible conflict poolout and classifier.
mopts.poolType='compact2';
mopts.learnW=false;
mopts.projDim=8192;
gpuId=1;
tag='fewshot';
saveInter=2;

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% few shot bilinear
mopts.poolType='bilinear';
mopts.learnW=false;
mopts.projDim=8192;
gpuId=1;
tag='fewshot';
saveInter=2;
mopts.classifyType='SVM';

run_one_experiment(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);