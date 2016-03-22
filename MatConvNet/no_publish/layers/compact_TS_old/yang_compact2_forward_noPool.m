function now=yang_compact2_forward_noPool(layer, pre, now)
    bsn=8; % size of image batch
    
    ch=layer.h; % size 2*c 1-d
    if ~isfield(layer, 'weights')
        layer.weights={layer.s};
    end
    cs=layer.weights{1}; % size 2*c +-1
    outDim=layer.outDim;
    
    x=pre.x;  % approx size is 512=c
    [h,w,c,n]=size(x);
    x=permute(x, [3,1,2,4]); % order c h w n
    x=reshape(x, c, h*w*n);
    
    if isa(x, 'gpuArray')
        gpuMode=true;
    else
        gpuMode=false;
    end
    
    now.x=[];
    if gpuMode
        out=gpuArray.ones(outDim, h, w, n, 'single'); % approx size is 5000=projDim
    else
        out=ones(outDim, h, w, n, 'single');
    end
    
    for img=1:ceil(n/bsn)
        interLarge=getInter(img, bsn*h*w, n*h*w);
        interSmall=getInter(img, bsn, n);
        out(:, :,:, interSmall)=aBatch(ch, cs, x(:, interLarge), gpuMode, ...
            outDim, h, w, c, numel(interSmall), layer.ia, layer.comp);
    end
    
    now.x=permute(out, [2,3,1,4]);
end

function out=aBatch(ch, cs, x, gpuMode, outDim, h, w, c, n0, ia, comp)
    if gpuMode
        out=gpuArray.ones(outDim, h*w*n0, 'single'); % approx size is 5000=projDim
    else
        out=ones(outDim, h*w*n0, 'single');
    end
    
    for ipoly=1:2
        hi=ch(ipoly, :);
        si=cs(ipoly, :);
        
        count=getCount(hi, si, x, gpuMode, outDim, h, w, c, n0,...
            ia{ipoly}, comp{ipoly}); % approx size 5000=projDim
        count=fft(count, [], 1);
        out=out .* count;
    end
    out=real(ifft(out, [], 1));

    out=reshape(out, outDim, h, w, n0);
end