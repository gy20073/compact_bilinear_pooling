function now=yang_sqrt_forward(layer, pre, now)
    in=pre.x;
    now.x=sign(in).*sqrt(abs(in));
end