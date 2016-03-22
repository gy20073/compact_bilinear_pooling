% also only validating an assumption, not of practical usage
function [trainFs, testFs]=normalize_var1_linear(trainFs, testFs)
    all=[trainFs testFs];
    v=var(all, 0, 2);
    all=bsxfun(@times, all, 1./(sqrt(v)+1e-10));
    
    trainFs=all(:, 1:size(trainFs, 2));
    testFs=all(:, (size(trainFs,2)+1): end);
end
