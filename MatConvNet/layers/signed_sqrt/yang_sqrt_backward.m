function pre=yang_sqrt_backward(layer, pre, now)
    % should always set the ep LARGE enough to avoid gradient explosion
    ep=1e-1;
    pre.dzdx=now.dzdx .* 0.5 ./ (sqrt(abs(pre.x)) + ep);
end
