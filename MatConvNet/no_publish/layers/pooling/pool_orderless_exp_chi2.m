% mean pooling of exp_chi2 feature, doesn't work so well
% because of the summation nature instead of the product nature
function [trainFs, testFs]=pool_orderless_exp_chi2(trainFs, testFs, outDim)
    w=randn(outDim, size(trainFs, 3)*5, 'single');
    b=rand(outDim, 1, 'single')*2*pi;
    
    trainFs=oneBig(trainFs, outDim, w, b, true);
    testFs=oneBig(testFs, outDim, w, b, true);
end

function out=oneBig(big, outDim, w, b, useGPU)
    bs=800;
    N=size(big, 4);
    out=zeros(outDim, N, 'single');
    for i=1: ceil(N/bs)
        inter=((i-1)*bs+1): min(i*bs, N);
        inter1=inter(1:floor(numel(inter)/2));
        inter2=inter(floor(numel(inter)/2 + 1):end);
        bigSlice1=big(:,:,:,inter1);
        bigSlice2=big(:,:,:,inter2);
        if useGPU
            w=gpuArray(w);
            b=gpuArray(b);
            outDim=gpuArray(outDim);
            bigSlice1=gpuArray(bigSlice1);
            bigSlice2=gpuArray(bigSlice2);
        end
        [t1, t2]=...
            pool_exp_chi2_impl(bigSlice1, bigSlice2, outDim, w, b, useGPU);
        out(:, inter1)=gather(t1);
        out(:, inter2)=gather(t2);
    end
end

function [trainFs, testFs]=pool_exp_chi2_impl(trainFs, testFs, outDim, paramw, paramb, useGPU)
    [h,w,c,n]=size(trainFs);
    
    fprintf('chi2 projecting train\n')
    trainFs=chi2(trainFs, useGPU);
    fprintf('chi2 projecting test\n')
    testFs=chi2(testFs, useGPU);
    
    fprintf('RBF projecting \n')
    [trainFs, testFs]=feature_RBF_approximation(trainFs, testFs, outDim, paramw, paramb, useGPU);
    
    fprintf('mean pooling\n')
    trainFs=rsp(trainFs,h,w,c,n);
    testFs=rsp(testFs,h,w,c,n);
end

function fs=chi2(fs, useGPU)
    [h,w,c,n]=size(fs);
    fs=permute(fs, [3,1,2,4]);
    fs=reshape(fs, c, h*w*n);
    
    fs=proj_chi2(fs, 2, 0.5, useGPU);
end

function fs=rsp(fs,h,w,c,n)
    fs=reshape(fs, size(fs, 1), h*w, n);
    fs=squeeze(mean(fs, 2));
end
