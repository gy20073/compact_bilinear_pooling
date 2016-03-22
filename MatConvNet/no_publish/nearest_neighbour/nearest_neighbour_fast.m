% This function is actually not as fast as the _yang version on GPU
% and further, you could test it by subsampling. So I prefer to use
% the _yang version. 
function nearest_neighbour_fast(trainFV, trainY, valFV, valY)
    %%
    valFV=valFV_bak;
    valY=valY_bak;
    sampleFactor=10;
    
    [h,w,~,~]=size(trainFV);
    nclass=max(trainY);
    
    valFV=valFV(:,:,:,1:sampleFactor:end);
    ntest=size(valFV, 4);
    valFV=dim4todim2(valFV);
    
    valY=valY(:,:,:,1:sampleFactor:end);
    
    cls=zeros(ntest, nclass);
    
    libpath='/home/yang/local/my_software/flann-1.8.4-src/src/matlab';
    addpath(libpath);
    
    %for ic=1:nclass
    for ic=1:nclass
        fprintf('calculating distance for class %d\n', ic);
        cur=dim4todim2(trainFV(:,:,:,trainY==ic));
        
        %build_params.algorithm='kdtree';
        %build_params.trees=8;
        build_params.algorithm = 'autotuned';
        build_params.target_precision = 0.85;
        build_params.build_weight = 0.1;
        build_params.memory_weight = 0;
        
        flann_set_distance_type('euclidean');
        fprintf('building index\n');
        [index, parameters, speedup]=flann_build_index(cur, build_params);
        speedup
        
        fprintf('performing NN search\n');
        [results, dists]=flann_search(index, valFV, 1, parameters);
        % dists is of size h*w*ntest
        dists=reshape(dists, h*w, ntest);
        cls(:, ic)=sum(dists, 1);
        flann_free_index(index);
    end
    
    fprintf('using nearst neighbour: accuracy is %f, mAP is %f\n', ...
        score_accuracy(cls, valY), ...
        mean(score_mAP(cls, valY)));
end
