function out=dist_bilinear(a, b)
    [a, b]=pool_bilinear(a, b, true);
    out=dot(a, b, 1);
end
