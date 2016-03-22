% normalize the whole image blob, not each pixel individually
function [trainFs, testFs]=normalize_l1_instance(trainFs, testFs)
    trainFs=one(trainFs);
    testFs=one(testFs);
end

function fs=one(fs)
    [h,w,c,n]=size(fs);    
    ss=sum(reshape(fs, h*w*c, n));
    ss=reshape(ss, 1,1,1,n);
    fs=bsxfun(@rdivide, fs, ss);
    
    % optional: scale each vector to a l1==1 scale
    fs=fs*(h*w);
end

