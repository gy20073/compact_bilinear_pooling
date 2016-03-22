function [trainFs, testFs]=pool_orderless(trainFs, testFs, str_method)
    eval(['method=@' str_method ';']);

    trainFs=method(trainFs);
    testFs=method(testFs);
end

function fs=x_x2(fs)
    first=squeeze(mean(mean(fs)));
    second=squeeze(mean(mean(fs.^2)));
    fs=cat(1, first, second);
end

function fs=mpool(fs)
    fs=squeeze(mean(mean(fs)));
end

function fs=chi2(fs)
    fs=proj_chi2_mean(fs, 10, 0.2, true);
end
