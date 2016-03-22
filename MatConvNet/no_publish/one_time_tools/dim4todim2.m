function trainFV_fs=dim4todim2(trainFV)
    [h,w,c,ntrain]=size(trainFV);
    trainFV_fs=permute(trainFV, [3,1,2,4]);
    trainFV_fs=reshape(trainFV_fs, c, h*w*ntrain);
end
