%% this file extract part locations, features and do classifications
%% read the part file
file_part='./data/cub/parts/part_locs.txt';
file_partnames='./data/cub/parts/parts.txt';
[imageid, partid, x, y, vis]=textread(file_part, '%d %d %f %f %d');
[~, strnames]=textread(file_partnames, '%d %s', 'whitespace', '\n');

% the structure of parts: ([x, y, isVis], partid, imageId)
parts=zeros(3, 15, 11788);
for i=1:length(imageid)
    parts(:, partid(i), imageid(i))=[x(i), y(i), vis(i)];
end

%%
imdb=load(consts('CUB', 'imdb'));

% read through image sizes and transform parts size to reshaped size
hw=zeros(2, 11788);
for i=1:length(imdb.images.name)
    iname=[imdb.imageDir '/' imdb.images.name{i}];
    tmp=imfinfo(iname);
    hw(:, i)=[tmp.Width, tmp.Height];
end

imdb.images.parts=parts;
imdb.images.hw=hw;
save(consts('CUB','parts'), 'imdb');
%% 
load(consts('CUB','parts'));
% visualize some image
id=3; % input, also the imdb
imname=[imdb.imageDir '/' imdb.images.name{id}];
part_showKeyPoints(imname, imdb.images.parts(:,:,id));

%% convert part location in terms of the reshaped image
%   :explore with the getbatch reshape
getbatch=getBatchWrapper(struct('numThreads', 4,...
                        'imageSize', [448 448 3], 'averageImage', [0 0 0]));
id=3;
a=getbatch(imdb, id);
a=uint8(a);

part_showKeyPoints(a, ...
  part_original2reshape(imdb.images.parts(:,:,id), imdb.images.hw(:,id)));

%% simplify the getbatch reshape
b=imread(imname);
factor=max((448)/size(b,1), (448)/size(b,2));
b=imresize(b, factor);
start=size(b);
start=ceil((start(1:2)+1-448)/2);
b=b(start(1):(start(1)+448-1), start(2):(start(2)+448-1), :);
imshow(b)

%% have a map from original axis to new axis
% part_original2reshape
% and apply to the imdb structure
parts_transformed=zeros(3, 15, 11788);
for i=1: length(imdb.images.id)
    parts_transformed(:,:,i)= part_original2reshape(...
            imdb.images.parts(:,:,i), imdb.images.hw(:,i));
end
imdb.images.parts_transformed=parts_transformed;

save(consts('CUB','parts'), 'imdb');
%% have a map from reshaped image to feature map
% look at fast RCNN
% decided to directly rescale to the target size
%% recall the structure of the saved feature
fpath=consts('CUB', 'poolout', 'network', 'VGG_16', ...
             'cfgId', 'compact2_finetuned', ...
             'use448', true, 'projDim', 8192);
load(fpath);         

% this accuracy is 82%, map=84.8%, a little less than the finetuned
train_test_vlfeat('LR', trainFV, trainY, valFV, valY);
%% 
%getIntermediateActivations
% then get the file:
% /home/yang/exp_data/fv_layer/exp_yang_fv/poolout_CUB_VGG_16_conv5_relu_448.mat
%% extract parts feature
[trainFV_part, valFV_part]=part_get_feature(trainFV, valFV, imdb);
outpath='/home/yang/exp_data/fv_layer/exp_yang_fv/poolout_CUB_VGG_16_conv5_relu_parts_448.mat';
savefast(outpath,'trainFV_part','trainY','valFV_part','valY');
%% try some classification, on all parts, only head, etc.
getInter=@(ibatch, batchSize, upper) ...
    ((ibatch-1)*batchSize+1): min(upper, ibatch*batchSize);
trainFV_head=trainFV_part(getInter(6, 8192, +inf), :);
valFV_head=valFV_part(getInter(6, 8192, +inf), :);
train_test_vlfeat('LR', trainFV_head, trainY, valFV_head, valY);

%% some visualizations of extracted parts