% abandoned
function out=memSavingFFT(count, isInverse)
    bs=4; % batch size in megabytes
    
    
    [dim, n]=size(count);
    if isa(count, 'gpuArray')
        out=gpuArray.zeros(dim, n, 'single');
    else
        out=zeros(dim, n, 'single');
    end
    
    real_bs=floor(bs*1024*1024/dim/4); % assume single data type
    for i=1:ceil(n/real_bs)
        inter=getInter(i, real_bs, n);
        if isInverse
            out(: , inter)=ifft(count(:, inter), [], 1);
        else
            out(: , inter)=fft(count(:, inter), [], 1);
        end
    end
end
