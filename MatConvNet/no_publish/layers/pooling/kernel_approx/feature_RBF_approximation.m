function [trainFs, testFs]=feature_RBF_approximation(trainFs, testFs, outDim, w, b, useGPU)
    [odim, nsample]=size(trainFs);
    if nargin<=5
        useGPU=false;
    end
    if nargin<=3
        if ~useGPU
            w=randn([outDim, odim], 'single');
            b=rand([outDim, 1], 'single')*2*pi;
        else
            w=randn([outDim, odim], 'single', 'gpuArray');
            b=rand([outDim, 1], 'single', 'gpuArray')*2*pi;
        end
        %b=0;
    end

    proj=@(x,w,b,outDim)cos(bsxfun(@plus, w*x, b))*sqrt(2/outDim);
    fprintf('RBF projecting train\n')
    trainFs=proj(trainFs,w,b,outDim);
    
    fprintf('RBF projecting test\n')
    testFs=proj(testFs,w,b,outDim);
end
