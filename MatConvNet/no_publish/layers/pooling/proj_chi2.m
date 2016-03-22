function out=proj_chi2(fs, n, L, useGPU, checkOn)
    if nargin<=4
        checkOn=false;
    end
    if nargin<=3
        useGPU=false;
    end
    
    bs=10000;
    [dim, nsample]=size(fs);
    out=zeros(dim*(2*n+1), nsample, 'single');
    for i=1:ceil(nsample/bs)
        inter=((i-1)*bs+1):min(i*bs, nsample);
        temp=fs(:, inter);
        if useGPU
            temp=gpuArray(temp);
        end
        out(:, inter)=gather(proj_chi2_impl(temp, n, L, useGPU));
    end
    
    %% sanity check to make sure that projection is accurate
    if checkOn
        sps=randsample(size(fs, 2), 2000);
        sp1=sps(1:1000);
        sp2=sps(1001:2000);
        t1=fs(:, sp1);
        t2=fs(:, sp2);
        p1=out(:, sp1);
        p2=out(:, sp2);

        test_chi2_sub(t1, t2, p1, p2);
    end
end

function out=proj_chi2_impl(fs, n, L, useGPU)
    if nargin<=3
        useGPU=false;
    end

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
end

function out=getInter(iseg, lenSingle)
    out=((iseg-1)*lenSingle+1) : (iseg*lenSingle);
end

function logx=safeLog(x)
    logx=max(log(x), -15);
end