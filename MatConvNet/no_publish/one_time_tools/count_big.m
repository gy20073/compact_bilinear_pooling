function out=count_big(a, thres)
    out=sum(abs(a(:))>thres)/numel(a);
end