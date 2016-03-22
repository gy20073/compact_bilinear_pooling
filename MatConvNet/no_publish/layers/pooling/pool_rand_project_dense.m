function [trainFs, testFs]=pool_rand_project_dense(trainFs, testFs, outDim)
    tic
    fprintf('random projection: ')
    a=cat(2, trainFs, testFs);
    std=sqrt(1.0/outDim);
    mx=randn(outDim, size(a, 1), 'single')*std;
    a=mx*a;

    ntrain=size(trainFs, 2);
    trainFs=a(:, 1:ntrain);
    testFs=a(:, (ntrain+1):end);
    toc
end

% deprecated function because: sum of linear transformation == linear
% transformation of sum (average pooling)
function fs=one(fs, projm)
    [outDim, ~]=size(projm);
    [h,w,c,n]=size(fs);
    fs=permute(fs, [3,1,2,4]); % c h w n
    hwn=h*w*n;
    fs=reshape(fs, c, hwn);
    fs=projm*fs;d
    fs=reshape(outDim, h, w, n);
    fs=squeeze(mean(mean(fs, 2),3));
end

