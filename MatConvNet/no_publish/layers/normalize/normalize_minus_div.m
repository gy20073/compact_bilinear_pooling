function [trainFs, testFs]=normalize_minus_div(trainFs, testFs, isMinus0)
    % calculates the normalization method
    fmax=max(trainFs, [], 2);
    fmin=min(trainFs, [], 2);

    M=zeros(size(trainFs, 1), 2);
    M(:, 1)=mean(trainFs, 2);
    if isMinus0
        M(:, 1)=0;
    end
    M(:, 2)=fmax-fmin+1e-10;

    trainFs=bsxfun(@minus, trainFs, M(:, 1));
    trainFs=bsxfun(@rdivide, trainFs, M(:, 2));
        
    testFs=bsxfun(@minus, testFs, M(:, 1));
    testFs=bsxfun(@rdivide, testFs, M(:, 2));
end
