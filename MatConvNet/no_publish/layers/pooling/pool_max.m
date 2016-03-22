function [trainFs, testFs]=pool_max(trainFs, testFs)
    trainFs=one(trainFs);
    testFs=one(testFs);
end

function fs=one(fs)
    fs=max(max(fs));
    fs=squeeze(fs);
end