function [trainFs, testFs]=normalize_rank(trainFs, testFs)
    all=[trainFs testFs];
    [~, I]=sort(all, 2);
    for i=1:size(all, 1)
        all(i, I(i, :))=1:size(all, 2);
    end
    
    trainFs=all(:, 1:size(trainFs, 2));
    testFs=all(:, (size(trainFs,2)+1): end);
end