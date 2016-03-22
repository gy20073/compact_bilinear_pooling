%%
setup_yang
%%
use448=true;
batchSize=128;
[trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer('MIT', 'VGG_M', 13, use448, [], batchSize);
[trainFV, valFV]=relu(trainFV, valFV);
%% average pooling
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_ave, valFV_ave);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);

%% get the pyramid features
[trainFV_pyramid, valFV_pyramid]=pool_pyramid(trainFV, valFV);
%%
trainFV_pyramid2=trainFV_pyramid(1:8192, :); valFV_pyramid2=valFV_pyramid(1:8192,:);
[trainFV_root, valFV_root]=normalize_root(trainFV_pyramid2, valFV_pyramid2);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);
%% two resize pyramid
indDim1=1:8192*5;
indDim2=1:8192*5;

a=load('data/exp_yang_fv/mit_pyramid224.mat');
trainFV_1=a.trainFV_pyramid(indDim1,:);
valFV_1=a.valFV_pyramid(indDim1, :);

b=load('data/exp_yang_fv/mit_pyramid448.mat');
trainFV_2=b.trainFV_pyramid(indDim2,:);
valFV_2=b.valFV_pyramid(indDim2, :);
%% 1
trainFV_pyramid2=cat(1, trainFV_1, trainFV_2);
valFV_pyramid2=cat(1, valFV_1, valFV_2);
%% 2
trainFV_pyramid2=trainFV_2;
valFV_pyramid2=valFV_2;
%% 3
trainFV_pyramid2=trainFV_1+trainFV_2;
valFV_pyramid2=valFV_1+valFV_2;
%%
trainY=a.trainY;
valY=a.valY;
%% classify
[trainFV_root, valFV_root]=normalize_root(trainFV_pyramid2, valFV_pyramid2);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);

%% run an bilinear baseline
use448=true;
batchSize=128;
[trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer('MIT', 'VGG_M', 13, use448, [], batchSize);
[trainFV, valFV]=relu(trainFV, valFV);
%%
[trainFV_bilinear, valFV_bilinear]=pool_bilinear(trainFV, valFV, true); 
[trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);

%% extract multi scale features
scales=[120, 170, 224, 280, 360, 448, 600];
%scales=1024;
gpuIds=[1 2 4 8];
dataset='MIT';
network='VGG_16';
gpuDevice(3)

for scale=scales
    batchSize=2;
    [trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer(...
        dataset, network, consts(dataset, 'endLayer', 'network', network), double(scale), [], batchSize);
    [trainFV, valFV]=relu(trainFV, valFV);
    %global h s outDim 
    %h=[]; s=[]; outDim=[];
    [trainFV_pyramid, valFV_pyramid]=pool_pyramid(trainFV, valFV);
    %s(1:10)
    fpath=['data/exp_yang_fv/' dataset '_' network '_pyramid_' num2str(scale) '.mat'];
    savefast(fpath,'trainFV_pyramid', 'valFV_pyramid', 'trainY','valY')
end
%% parallel one don't work
numWorker=numel(gpuIds);
spmd(numWorker)
    gId=gpuIds(labindex);
    gpuDevice(gId);
    
    for scale=scales(labindex: numWorker: end)
        [gId, scale]
        
        batchSize=floor(8*448*448/scale/scale);
        [trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer(...
            dataset, network, consts(dataset, 'endLayer', 'network', network), double(scale), [], batchSize);
        [trainFV, valFV]=relu(trainFV, valFV);
        
        [trainFV_pyramid, valFV_pyramid]=pool_pyramid(trainFV, valFV);
        fpath=['data/exp_yang_fv/' dataset '_' network '_pyramid_' num2str(scale) '.mat'];
        savefast(fpath,'trainFV_pyramid', 'valFV_pyramid', 'trainY','valY')
    end
end

%% pyramid classification

scales=[120, 170, 224, 280, 360, 448, 600];
scales=[360, 448, 600];
dataset='MIT';
network='VGG_16';
isConcat=1;

iscale=0;
clear trainFV
for scale=scales
    iscale=iscale+1;
    fpath=['data/exp_yang_fv/' dataset '_' network '_pyramid_' num2str(scale) '.mat']
    load(fpath,'trainFV_pyramid', 'valFV_pyramid', 'trainY','valY')
    if ~exist('trainFV')
        [dim, n1]=size(trainFV_pyramid);
        [dim2, n2]=size(valFV_pyramid);
        assert(dim==dim2);
        if isConcat
            trainFV=zeros(dim*numel(scales), n1,'single');
            valFV=zeros(dim*numel(scales), n2,'single');
        else
            % this is sum concatenation
            trainFV=zeros(dim, n1,'single');
            valFV=zeros(dim, n2,'single');
        end
    end
    if isConcat
        trainFV(getInter(iscale, dim, inf), :)=trainFV_pyramid;
        valFV(getInter(iscale, dim, inf), :)=valFV_pyramid;
    else
        % try normalize first then add
        [trainFV_root, valFV_root]=normalize_root(trainFV_pyramid, valFV_pyramid);
        [trainFV_pyramid, valFV_pyramid]=normalize_l2(trainFV_root, valFV_root);
        % add
        trainFV=trainFV+trainFV_pyramid;
        valFV=valFV+valFV_pyramid;
        % compatable with classification
        trainFV_l2=trainFV;
        valFV_l2=valFV;
        %[trainFV_l2, valFV_l2]=normalize_root(trainFV_l2, valFV_l2);
        [trainFV_l2, valFV_l2]=normalize_l2(trainFV_l2, valFV_l2);
    end
end
%%
trainFV2=trainFV; valFV2=valFV;
%% classify
[trainFV_root, valFV_root]=normalize_root(trainFV2, valFV2);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
%%  
train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);
%%
inter=getInter(5, 8192, inf);
trainFV2=trainFV(inter,:);
valFV2=valFV(inter, :);
%%
