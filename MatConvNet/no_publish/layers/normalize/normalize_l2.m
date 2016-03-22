function [trainFs, testFs]=normalize_l2(trainFs, testFs)
    % l2 normalize the features
    l2norm=@(x)single( bsxfun(@rdivide, x, sqrt(sum(x.^2, 1))) );
    trainFs=l2norm(trainFs);
    testFs=l2norm(testFs);
end
