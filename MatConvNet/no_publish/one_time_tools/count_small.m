function out=count_small(a, thres)
    out=sum(abs(a(:))<thres)/numel(a);
end