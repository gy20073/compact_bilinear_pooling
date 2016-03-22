function now=yang_compact_bilinear_RM_forward(layer, pre, now)
    now=yang_compact_bilinear_RM_forward_nopool(layer, pre, now);
    now.x=sum(sum(now.x, 1), 2);
end
