function [trainFs, testFs]=normalize_sparse(trainFs, testFs, threshold)
    tosp=@(x)x-x.*(abs(x)<threshold);
    trainFs=tosp(double(trainFs));
    testFs=tosp(double(testFs));
end

