% only a test of the formula, not of practical usage
function [trainFs, testFs]=normalize_taylor_1st(trainFs, testFs)
    all=[trainFs testFs];
    ave=mean(all, 2);
    x_u=bsxfun(@minus, all, ave);
    second=bsxfun(@times, x_u, sign_root_der(ave) );
    all=bsxfun(@plus, second, sign_root(ave));
    
    trainFs=all(:, 1:size(trainFs, 2));
    testFs=all(:, (size(trainFs,2)+1): end);
end

function out=sign_root(a)
    out=sqrt(abs(a)).*sign(a);
end

function out=sign_root_der(a)
    out=0.5./sqrt(abs(a)+1e-16);
end
