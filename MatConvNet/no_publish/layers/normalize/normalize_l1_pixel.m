% normalize each pixel individually
function [trainFs, testFs]=normalize_l1_pixel(trainFs, testFs)
    trainFs=one(trainFs);
    testFs=one(testFs);
end

function fs=one(fs)
    ss=sum(fs, 3);
    fs=bsxfun(@rdivide, fs, ss+(1e-10));
    
    % optional
    %fs=fs*10;
end
