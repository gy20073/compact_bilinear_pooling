function [trainFs, testFs]=feature_poly_approximation(trainFs, testFs, outDim)
    [h,w,c,n]=size(trainFs);
    ws=cell(1,2);
    for i=1:2
        ws{i}=(randi(2, outDim, c)*2-3);
    end
    
    if ~isempty(trainFs)
        trainFs=one(trainFs, ws, outDim, true);
    end
    if ~isempty(testFs)
        testFs=one(testFs, ws, outDim, true);
    end
end

function out=one(fs, ws, outDim, useGPU)
    [h,w,c,n]=size(fs);
    out=zeros(outDim, n, 'single');
    if useGPU
        out=gpuArray(out);
    end
    fs=permute(fs, [3,1,2,4]); % ordering: c h w n
    fs=reshape(fs, c, h*w*n);
    
    bs=250;
    for i=1:ceil(n/bs)
        fprintf('batch id=%d\n', i);
        iinter=getInter(i, bs*h*w, n*h*w);
        cur=fs(:, iinter);
        if useGPU
            cur=gpuArray(cur);
        end
        temp=(ws{1}*cur) .* (ws{2}*cur); % size: outDim*(n*h*w)
        temp=reshape(temp, outDim, h*w, []);
        out(:, getInter(i, bs, n))=squeeze(mean(temp, 2));
    end
    out=out/sqrt(outDim);
    out=gather(out);
end

function out=getInter(iseg, len, upperBound)
    out=((iseg-1)*len+1):min(iseg*len, upperBound);
end
