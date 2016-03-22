function [trainFs, testFs]=dim_reduction_random(trainFs, testFs, outDim)
    ndim=size(trainFs, 1);
    ind=randsample(ndim, outDim);

    trainFs=trainFs(ind, :);
    testFs=testFs(ind, :);
end

