% not working so well
function [trainFs, testFs]=pool_pyramid(trainFs, testFs)
    trainFs=one(trainFs);
    if ~isempty(testFs) 
        testFs=one(testFs);
    end
end

function out=one(fs)
    method=@(fs) poly2(fs);
    outDim=8192;
    bs=5000;
    
    scales=[1];
    weight=[1/4 1/4 1/2];
    
    [h,w,c,n]=size(fs);
    out=zeros(outDim*sum(scales.^2), n, 'single');
    getInter=@(i, c, upper) (c*(i-1)+1) : min(c*i, upper);
    
    % multi scale pooling
    nextLoc=1; iscale=0;
    for scale=scales
        iscale=iscale+1;
        for i=1:scale
            stepi=floor(h/scale);
            for j=1:scale
                % batching
                for ibatch=1:ceil(n/bs)
                    batchInter=getInter(ibatch, bs, n);
        stepj=floor(w/scale);
        cur=fs(getInter(i, stepi, h), ...
               getInter(j, stepj, w), :,batchInter);
        out(getInter(nextLoc, outDim, inf),batchInter)=weight(iscale)*method(cur);
        nextLoc=nextLoc+1;
                end
            end
        end
    end
end

function out=mpool(fs)
    out=squeeze(mean(mean(fs)));
end

function out=poly2(fs)
    global h s outDim
    if isempty(h) || isempty(s)
        'getting new projection param, should be here for only once'
        outDim=8192;
        [~,~,c,~]=size(fs);
        h=randi(outDim, 2, c);
        s=randi(2,2,c)*2-3;
    end
    
    [out, ~]=feature_poly2_approximation(fs, [], outDim, h,s); 
end