function count=getCount(hi, si, x, gpuMode, outDim, h, w, c, n, ia, comp)
    count=getCount1(hi, si, x, gpuMode, outDim, h, w, c, n, ia, comp);
end

function count=getCount1(hi, si, x, gpuMode, outDim, h, w, c, n, ia, comp)
    if gpuMode
        count=gpuArray.zeros(outDim, h*w*n, 'single');
    else
        count=zeros(outDim, h*w*n, 'single');
    end

    count(hi(ia), :)=bsxfun(@times, si(ia)', x(ia, :));
    % for the rest iterate to process
    for i=comp
        count(hi(i), :)=count(hi(i), :) + si(i)*x(i,:);
    end 
end

function count=getCount2(hi, si, x, gpuMode, outDim, h, w, c, n)
    if gpuMode
        count=zeros([outDim, h*w*n], 'single', 'gpuArray');
    else
        count=zeros(outDim, h*w*n, 'single');
    end
    %for each dimension of the original features
    for idim=1:c
        count(hi(idim), :)=count(hi(idim), :)+si(idim)*x(idim, :);
    end
end