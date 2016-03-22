function out=dist_chi2(a, b)
    % a and b are both feature matrix
    div=2*(a.*b)./(a+b+1e-20);
    out=sum(div, 1);
end