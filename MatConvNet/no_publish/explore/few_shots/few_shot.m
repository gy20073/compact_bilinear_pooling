function few_shot(trainFV, trainY, valFV, valY)
    gpuDevice(2);
    %classifier='SVM';
    classifier='LR';
    
    outName=['data/exp_yang_fv/few_shot_res_' classifier '.mat'];
    % default values for input
    setup_yang
    if nargin<4
        dataset='CUB';
        network='VGG_M';
        endLayer=consts(dataset, 'endLayer', 'network', network);
        % load the CUB activations
        [trainFV, trainY, valFV, valY]=get_activations_dataset_network_layer(dataset, network, endLayer);
        [trainFV, valFV]=relu(trainFV, valFV);
    end
   
    % train and test
    dims=2.^[5 7 9 11 13 14 15];
    fewShots=[1, 2, 3, 4, 5, 7, 9, 11, 14];
    methods={'bilinear', 'compact1', 'compact2'};
    
    if exist(outName, 'file')==2
        load(outName);
    else
        out=[];
    end
    
    for dim=dims
        strDim=['n' num2str(dim)];
        for shot=fewShots
            strShot=['n' num2str(shot)];
            
            
            changed=false;
            for imethod=1:numel(methods)
                method=methods{imethod};
                try
                    temp=out.(strDim).(strShot).(method);
                    fprintf('done skipping dim=%d, shot=%d, method=%s\n', dim, shot, method);
                catch
                    if strcmp(method, 'bilinear')
                        try
                            temp=out.n8192.(strShot).bilinear;
                            fprintf('skipping bilinear, shot=%d\n', shot);
                            continue;
                        catch
                            % don't have bilinear, go on
                        end
                    end
                    if ~changed
                        [fv, y]=subsample(trainFV, trainY, shot);
                        changed=true;
                    end
                    
                    [acc, map, cls, out]=test_one_cache(fv, y, valFV, valY, dim, method, out);
                    out.(strDim).(strShot).(method)={acc, map, cls};
                    fprintf('testing proj dim=%d, shot=%d, method=%s, acc=%f, mAP=%f\n',...
                        dim, shot, method, acc, map);
                end
            end
            if changed
                savefast(outName, 'out');
            end
        end
    end
end

function [fv, y]=subsample(trainFV, trainY, n)
    % make sure generate the same set for a single n
    rng('default');

    nclass=max(trainY);
    fv=[];
    y=[];
    
    for iclass=1:nclass
        cind=(trainY==iclass);
        tTrain=trainFV(:,:,:,cind);
        tY=trainY(cind);
        
        sz=numel(tY);
        ind=randsample(sz, n);
    
        fv=cat(4, fv, tTrain(:,:,:, ind));
        y=cat(1, y, tY(ind));
    end
end

function [acc, map, cls]=test_one_notefficient(trainFV, trainY, valFV, valY, dim, method)
    compact1=@(trainFV, valFV, projDim)feature_poly_approximation(trainFV, valFV, projDim);
    compact2=@(trainFV, valFV, projDim)feature_poly2_approximation(trainFV, valFV, projDim);
    bilinear=@(trainFV, valFV, projDim)pool_bilinear(trainFV, valFV, true); 

    rng(1);
    eval(['[trainFV_bilinear, valFV_bilinear]=' method '(trainFV, valFV, dim);']);
    [trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
    [trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
    [~,~, acc, map, cls]=train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);
end

function [acc, map, cls, out]=test_one_cache(trainFV, trainY, valFV, valY, dim, method, out)
    compact1=@(trainFV, valFV, projDim)feature_poly_approximation(trainFV, valFV, projDim);
    compact2=@(trainFV, valFV, projDim)feature_poly2_approximation(trainFV, valFV, projDim);
    bilinear=@(trainFV, valFV, projDim)pool_bilinear(trainFV, valFV, true); 
    
    if strcmp(method, 'bilinear')
        cname='cache_bilinear';
    else
        cname=['cache_' method '_' num2str(dim)];
    end
    
    rng('default');
    if ~isfield(out, cname) || isempty(out.(cname))
        fprintf('not fount: %s', cname);
        % no cache found, operates as usual
        eval(['[trainFV_bilinear, valFV_bilinear]=' method '(trainFV, valFV, dim);']);
        [trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
        [trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
        % save the output var
        if ~strcmp(method, 'bilinear')
            out.(cname)=valFV_l2;
        end
    else
        fprintf('using the cached: %s', cname);
        % cache found, return that half
        eval(['[trainFV_bilinear, valFV_bilinear]=' method '(trainFV, [], dim);']);
        [trainFV_root, valFV_root]=normalize_root(trainFV_bilinear, valFV_bilinear);
        [trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
        % save the output var
        valFV_l2=out.(cname);
    end
    [~,~, acc, map, cls]=train_test_vlfeat('SVM', trainFV_l2, trainY, valFV_l2, valY);
end
