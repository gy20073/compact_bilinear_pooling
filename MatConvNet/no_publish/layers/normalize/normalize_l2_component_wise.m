function [trainFs, testFs]=normalize_l2_component_wise(trainFs, testFs, compoLen)
    trainFs=one(trainFs, compoLen);
    testFs=one(testFs, compoLen);
end

function fs=one(fs, compoLen)
    l2norm=@(x)bsxfun(@rdivide, x, sqrt(sum(x.^2, 1)));
    assert( mod(size(fs, 1), compoLen)==0);
    for i=1:(size(fs,1)/compoLen)
        inter=(i-1)*compoLen+1 : i*compoLen;
        fs(inter,:)=l2norm(fs(inter,:));
    end
end