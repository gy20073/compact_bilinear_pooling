function [trainFs, testFs]=pool_ave(trainFs, testFs)
    trainFs=one(trainFs);
    testFs=one(testFs);
end

function fs=one(fs)
    fs=mean(mean(fs));
    fs=squeeze(fs);
end