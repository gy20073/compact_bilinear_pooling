function pre=yang_bilinear_backward(layer, pre, now)
    pre.dzdx=vl_nnbilinearpool(pre.x, now.dzdx);
end