function [trainFs, testFs]=feature_poly2_approximation(trainFs, testFs, outDim, h, s)
    order=2;
    
    if nargin<=3
        hs=cell(1,order);
        ss=cell(1,order);
        [h,w,c,n]=size(trainFs);
        for i=1:order
            hs{i}=randi(outDim, 1, c);
            ss{i}=randi(2, 1, c)*2-3;
        end
    else
        hs={h(1, :), h(2, :)};
        ss={s(1, :), s(2, :)};
    end
    
    if ~isempty(trainFs)
        trainFs=batchOne(trainFs, outDim, hs, ss, false);
    end
    if ~isempty(testFs)
        testFs=batchOne(testFs, outDim, hs, ss, false);
    end
end

function out=batchOne(fs, outDim, hs, ss, useGPU)
    bs=256;
    [h,w,c,n]=size(fs);
    out=ones(outDim, n, 'single');
    for i=1:ceil(n/bs)
        fprintf('batch id=%d\n', i);
        iinter=getInter(i, bs, n);
        out(:, iinter)=one(fs(:,:,:,iinter), outDim, hs, ss, useGPU);
    end
end

function out=getInter(iseg, len, upperBound)
    out=((iseg-1)*len+1):min(iseg*len, upperBound);
end

function counts=one(fs, outDim, hs, ss, useGPU)    
    if useGPU
        fs=gpuArray(fs);
    end
        
    [h,w,c,n]=size(fs);
    fs=permute(fs, [3,1,2,4]);
    fs=reshape(fs, c, h*w*n);
    
    if useGPU
        counts=ones([outDim, h*w*n], 'single', 'gpuArray');
    else
        counts=ones(outDim, h*w*n, 'single');
    end
    % for each polynomial
    for ipoly=1:numel(hs)
        if useGPU
            count=zeros([outDim, h*w*n], 'single', 'gpuArray');
        else
            count=zeros(outDim, h*w*n, 'single');
        end
            
        hi=hs{ipoly};
        si=ss{ipoly};
        %for each dimension of the original features
        for idim=1:c
            count(hi(idim), :)=count(hi(idim), :)+si(idim)*fs(idim, :);
        end
        counts=counts .* fft(count, [], 1);
    end
    counts=ifft(counts, [], 1);
    counts=reshape(counts, outDim, h*w, n);
    counts=squeeze(mean(counts, 2));
    if useGPU
        counts=gather(counts);
    end
end
