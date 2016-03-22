function now=yang_compact_bilinear_RM_forward_nopool(layer, pre, now)
    % w1 and w2 is size [ch, outDim]
    w1=layer.weights{1};
    w2=layer.weights{2};
    [nch, nout]=size(w1);
    [h,w,c,n]=size(pre.x);
    
    out=zeros([nout, n], 'like', pre.x);
    
    xin=permute(pre.x, [1,2,4,3]); % size is h, w, n, c
    xin=reshape(xin, h*w*n, c);    
    xin=(xin*w1).*(xin*w2);
    xin=reshape(xin, h, w, n, nout);
    
    now.x=permute(xin, [1 2 4 3]);

    % we attempted to replace two matrix multiplication with
    % a single, to improve the speed. But the speed is slower 
    % with the code below. 
    %xin=xin*cat(2, w1, w2);
    %xin=xin(:, 1:nout) .* xin(:, (nout+1):end);
    % replacement ends
end
