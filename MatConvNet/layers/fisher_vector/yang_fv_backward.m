function pre=yang_fv_backward(~, pre, now)
    pre.dzdx=yang_fv(pre.x, now.dzdx, now.x);
end