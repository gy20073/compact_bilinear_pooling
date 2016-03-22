function [trainFs, testFs]=relu(trainFs, testFs)
    trainFs=max(trainFs, 0);
    testFs=max(testFs, 0);
end
