function [trainFs, testFs]=normalize_l1(trainFs, testFs)
    l1norm=@(x)single( bsxfun(@rdivide, x, sum(abs(x), 1)) );
    trainFs=l1norm(trainFs);
    testFs=l1norm(testFs);
end