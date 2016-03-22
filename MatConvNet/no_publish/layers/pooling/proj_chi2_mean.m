function out=proj_chi2_mean(fs, n, L, useGPU)
    % convert from 4d blob to 2d
    [h,w,c,nsample0]=size(fs);
    fs=permute(fs, [3,1,2,4]);
    fs=reshape(fs, c, h*w*nsample0);
    % convert end
    
    if nargin<=3
        useGPU=false;
    end
    
    bs=h*w*500;
    bsOut=bs/h/w;
    [dim, nsample]=size(fs);
    out=zeros(dim*(2*n+1), nsample0, 'single');
    for i=1:ceil(nsample/bs)
        inter=((i-1)*bs+1):min(i*bs, nsample);
        interOut=((i-1)*bsOut+1):min(i*bsOut, nsample0);
        temp=fs(:, inter);
        if useGPU
            temp=gpuArray(temp);
        end
        out(:, interOut)=gather(proj_chi2_impl(temp, n, L, h, w, false, useGPU));
    end
    
    %% sanity check to make sure that projection is accurate
    sps=randsample(size(fs, 2), 2000);
    sp1=sps(1:1000);
    sp2=sps(1001:2000);
    t1=fs(:, sp1);
    t2=fs(:, sp2);
    p1=proj_chi2_impl(t1, n, L, h, w, true, useGPU);
    p2=proj_chi2_impl(t2, n, L, h, w, true, useGPU);
    
    test_chi2_sub(t1, t2, p1, p2);
end

function out=proj_chi2_impl(fs, n, L, h, w, noPool, useGPU)
    % fs is a feature_dim * nsamples matrix
    [dim, nsample]=size(fs);
    n21=2*n+1;
    if ~useGPU
        out=zeros(dim*n21, nsample, 'single');
    else
        out=zeros([dim*n21, nsample], 'single','gpuArray');
    end
    
    multi=sqrt(L)*sqrt(fs);
    out(getInter(1, dim), :)=multi;
    
    for k=1:n
        exp_pi_k_L=exp(pi*k*L);
        scaler=2/sqrt(exp_pi_k_L + 1/exp_pi_k_L);
        k_L_logx=k*L*safeLog(fs);
        
        % the sin part
        out(getInter(2*k, dim), :)=scaler * multi .* sin(k_L_logx);
        % the cos part
        out(getInter(2*k+1, dim), :)=scaler * multi .* cos(k_L_logx);
    end
    if ~noPool
        out=reshape(out, size(out,1), h*w, []);
        out=squeeze(mean(out, 2));
    end
end

function out=getInter(iseg, lenSingle)
    out=((iseg-1)*lenSingle+1) : (iseg*lenSingle);
end

function logx=safeLog(x)
    logx=max(log(x), -15);
end