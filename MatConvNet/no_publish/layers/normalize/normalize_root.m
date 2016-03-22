function [trainFs, testFs]=normalize_root(trainFs, testFs)
    sq=@(x)sqrt(abs(x)) .* sign(x);
    trainFs=sq(trainFs);
    testFs=sq(testFs);
end