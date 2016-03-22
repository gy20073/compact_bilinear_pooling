function getIntermediateActivations(netLoc, endLayer, dataset)
    %% some defauts
    %dataset='CUB_HEAD';
    %dataset='CUB_BODY';
    %detects={ 'CUB_BODY_DETECT', 'CUB_HEAD_DETECT'};
    %detects={'CUB'};
    %dataset='CUB';
    detects={'Marcel2'};
    
    now=detects;
    
    for idata=1:numel(now)
        dataset=now{idata};

        network='VGG_16';
        endLayer=33; % feature after compact2, sqrt, l2
        endLayer=30; % feature after relu after conv5_3
        hasModification=false;
        if strncmpi(dataset, 'CUB_BODY', 8)
            netLoc='/home/yang/exp_data/fv_layer/exp_yang_fv/CUB_BODY_VGG_16_LR_compact2_8192_learnW_224_ning/net-epoch-30.mat';
            use448=false;
        elseif strncmpi(dataset, 'CUB_HEAD', 8)
            netLoc='/home/yang/exp_data/fv_layer/exp_yang_fv/CUB_HEAD_VGG_16_LR_compact2_8192_learnW_224_ning/net-epoch-40.mat';
            use448=false;
        elseif strcmp(dataset, 'CUB')
            %netLoc='/home/yang/exp_data/fv_layer/exp_yang_fv/CUB_VGG_16_LR_compact2_8192_fixW_448_final_1/net-epoch-10.mat';
            use448=true;
        elseif strcmp(dataset, 'Marcel1')
            dataset='CUB';
            network='VGG_M';
            use448=true;
            endLayer=14; % relu5
            netLoc=consts('CUB', 'VGG_M');
            
            net=load(netLoc);
            net.layers=net.layers(1:endLayer);
            hasModification=true;
            % add a compact 2 layer to it
%             projDim=8192;
%             learnW=1;
%             TS_obj=yang_compact_bilinear_TS_nopool_2stream(projDim, [512 512], 1);
%             net.layers{end+1}=struct('type', 'custom',...
%                 'layerObj', TS_obj, ...
%                 'forward', @TS_obj.forward_simplenn_nopool, ...
%                 'backward', @TS_obj.backward_simplenn_nopool, ...
%                 'name', 'compact_TS', ...
%                 'outDim', projDim, ...
%                 'weights', {{cat(1, TS_obj.weights_{:})}}, ...
%                 'learningRate', [1]*learnW);
        elseif strcmp(dataset, 'Marcel2')
            dataset='CUB';
            network='VGG_M';
            use448=true;
            endLayer=14; % relu5
            netLoc='/home/yang/exp_data/fv_layer/exp_yang_fv/CUB_VGG_M_LR_compact_RM_8192_learnW_448_Marcel2_first100classes/net-epoch-67.mat';
        else
            error('no such dataset')
        end
        if ~hasModification
            a=load(netLoc);
            net=a.net;
            net.layers=net.layers(1:endLayer);
        end

        [trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer(...
        dataset, network, 'conv5_relu', use448, net, 64);

        %train_test_vlfeat('LR', trainFV, trainY, valFV, valY);
    end
end
