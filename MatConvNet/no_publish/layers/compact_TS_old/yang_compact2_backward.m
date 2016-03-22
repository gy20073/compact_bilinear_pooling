function pre=yang_compact2_backward(layer, pre, now)  
    bsn=8;
    
    ch=layer.h; % size 2*c 1-d
    if ~isfield(layer, 'weights')
        layer.weights={layer.s};
    end
    cs=layer.weights{1}; % size 2*c +-1
    outDim=layer.outDim;
    
    x=pre.x; % size 512
    [h,w,c,n]=size(x);
    x=permute(x, [3,1,2,4]); % order c h w n
    x=reshape(x, c, h*w*n);
    
    if isa(x, 'gpuArray')
        gpuMode=true;
    else
        gpuMode=false;
    end
    
    pre.dzdx=[];
    if gpuMode
        out=gpuArray.zeros(c, h*w*n, 'single'); % size 512
        gones=gpuArray.ones(1, h*w, 'single');
        dzdw=gpuArray.zeros(c, 1, 'single');
    else
        out=zeros(c, h*w*n, 'single');
        gones=ones(1, h*w, 'single');
        dzdw=zeros(c, 1, 'single');
    end
    dzdw={dzdw, dzdw};
    
    %% main calculation
    fftDzdy=fft(reshape(now.dzdx, outDim, n), [], 1); % size outDim*n
    
    for img=1:ceil(n/bsn)
        interLarge=getInter(img, bsn*h*w, n*h*w);
        interSmall=getInter(img, bsn, n);
        [out(:,interLarge), t_dzdw]=aBatch(fftDzdy(:, interSmall), gones, ch, cs, ...
            x(:, interLarge), gpuMode, outDim, h, w, c, numel(interSmall),...
            layer.ia, layer.comp, layer.learnW);
        if layer.learnW
            dzdw{1}=t_dzdw{1}+dzdw{1};
            dzdw{2}=t_dzdw{2}+dzdw{2};
        end
    end
    
    out=reshape(out, c, h*w, n);
    out=permute(out, [2,1,3]); % order hw, c, n
    pre.dzdx=reshape(out, h,w,c,n);
    if layer.learnW
        pre.dzdw={cat(2, dzdw{:})'};
    end
end

function [out, dzdw]=aBatch(fftDzdy, gones, ch, cs, x, gpuMode, outDim, h, w, c, n0, ia, comp, learnW)
    if gpuMode
        out=gpuArray.zeros(c, h*w*n0, 'single'); % size 512
    else
        out=zeros(c, h*w*n0, 'single');
    end
    dzdw=cell(1,2);
    
    repFftDzdy=kron(fftDzdy, gones); % size 5000
    
    for ipoly=1:2
        hi=ch(ipoly, :);
        si=cs(ipoly, :);
        count=getCount(hi, si, x, gpuMode, outDim, h, w, c, n0, ...
            ia{ipoly}, comp{ipoly}); % size 5000
        
        hip=ch(3-ipoly, :);
        sip=cs(3-ipoly, :);        
        count(2:end, :)=flipud(count(2:end, :));

        count=fft(count, [], 1);
        count=count .* repFftDzdy;
        dLdq=real(ifft(count ,[] ,1));
        
        out=out+bsxfun(@times, dLdq(hip,:), sip');
        if learnW
            dzdw{3-ipoly}=sum(dLdq(hip, :).*x,2);
        end
    end
end
