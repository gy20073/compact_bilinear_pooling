function [trainFs, testFs]=normalize_square(trainFs, testFs)
    sq=@(x) x.*x.*sign(x);
    trainFs=sq(trainFs);
    testFs=sq(testFs);
end