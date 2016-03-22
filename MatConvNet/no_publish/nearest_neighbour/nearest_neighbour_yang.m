function nearest_neighbour_yang(trainFV, trainY, valFV, valY, useGPU)
    %%
    useGPU=true;
    gpuDevice(8)
    
    trainY=gather(trainY);
    [h,w,c,ntrain]=size(trainFV);
    nclass=max(trainY);
    
    trainFV_fs=permute(trainFV, [3,1,2,4]);
    trainFV_fs=reshape(trainFV_fs, c, h*w*ntrain);
    if useGPU
        trainFV_fs=gpuArray(trainFV_fs);
    end
    
    ntest=size(valFV, 4);
    cls=zeros(ntest, nclass);
    
    
    sampleFactor=10;
    dist_l2=@(fs, di)sum(bsxfun(@minus, fs, di).^2, 1);
    dist_l1=@(fs, di)sum(abs(bsxfun(@minus, fs, di)), 1);
    dist_bilinear=@(fs, di)sum(bsxfun(@times, fs, di), 1).^2;
    dist_linear=@(fs, di)sum(bsxfun(@times, fs, di), 1);
    method=dist_l1;
    
    for itest=1:sampleFactor:ntest
        fprintf('testing the %d\n', itest);
        curImage=valFV(:,:,:,itest);
        if useGPU
            curImage=gpuArray(curImage);
        end
        for ih=1:h
            for iw=1:w
                di=curImage(ih, iw, :);
                di=reshape(di, c, 1);
                dist=method(trainFV_fs, di);
                % dist is length h*w*ntrain
                
                dist=gather(dist);
                trainY=gather(trainY);
                dist=reshape(dist, h*w, ntrain);
                for ic=1:nclass
                    temp=dist(:, trainY==ic);
                    cls(itest, ic)=cls(itest, ic)+min(temp(:));
                end
            end
        end
        [vt, it]=min(cls(itest, :));
        fprintf('label is %d, predicted is %d, is correct %d\n',...
            valY(itest), it, it==valY(itest))
    end
    
    %%
    if useGPU
        cls=gather(cls);
    end
    fprintf('using nearst neighbour: accuracy is %f, mAP is %f\n', ...
        score_accuracy(cls, valY), ...
        mean(score_mAP(cls, valY)));
end