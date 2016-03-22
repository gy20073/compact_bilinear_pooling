function pre=yang_compact_bilinear_RM_backward(layer, pre, now)
    [h, w, c, n]=size(pre.x);
    now.dzdx=repmat(now.dzdx, [h, w, 1, 1]);

    pre=yang_compact_bilinear_RM_backward_nopool(layer, pre, now);
    now.dzdx=[];
end
