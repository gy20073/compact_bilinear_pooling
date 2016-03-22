function now=yang_bilinear_forward(layer, pre, now)
    now.x=vl_nnbilinearpool(pre.x);
end