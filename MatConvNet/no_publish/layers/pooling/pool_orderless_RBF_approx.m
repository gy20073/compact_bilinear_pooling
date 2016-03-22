% a spatial RBF pooling kernel, adding
% not working well
function [trainFs, testFs]=pool_orderless_RBF_approx(trainFs, testFs, outDim)
    odim=size(trainFs, 3);
    w=randn(outDim, odim, 'single');
    b=rand(outDim, 1, 'single')*2*pi;
    
    trainFs=one(trainFs, w, b);
    testFs=one(testFs, w, b);
end

function fs=one(fs, W, B)
    [h,w,c,n]=size(fs);
    hwn=h*w*n;
    fs=permute(fs, [3,1,2,4]);
    fs=reshape(fs, c, hwn);

    % normalize root
    sq=@(x)sqrt(abs(x)) .* sign(x);
    %fs=sq(fs);
    % normalize l2
    l2norm=@(x)single( bsxfun(@rdivide, x, sqrt(sum(x.^2, 1))) );
    %fs=l2norm(fs);
    divNorm=@(x) x./ max(x(:))*5;
    fs=divNorm(fs);
    
    %fs=ave_proj(fs,h,w,c,n,W,B);
    fs=proj_pool(fs,h,w,c,n,W,B);

    % then l2 norm again
    fs=l2norm(fs);
end



function fs=ave_proj2(fs, W, B)
    fs=bsxfun(@plus, W*fs, B);
    fs=cos(fs);
end

function fs=ave_proj(fs,h,w,c,n, W, B)
    fs=reshape(fs, c, h*w, n);
    fs=squeeze(mean(fs, 2));
    
    fs=bsxfun(@plus, W*fs, B);
    fs=cos(fs);
end

function fs=proj_pool(fs, h,w,c,n,W,B)
    fs=bsxfun(@plus, W*fs, B);
    fs=cos(fs);
    
    outDim=size(W, 1);
    fs=reshape(fs, outDim, h*w, n);
    fs=squeeze(mean(fs, 2));
end
