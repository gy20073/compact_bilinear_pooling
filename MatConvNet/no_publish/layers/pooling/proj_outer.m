function [trainFV, testFV]=proj_outer(trainFV, testFV)
    trainFV=one(trainFV);
    testFV=one(testFV);
end

function out=one(fs)
    [dim, n]=size(fs);
    out=zeros(dim*dim, n,'single');
    for i=1:n
        out(:,i)=reshape(fs(:,i)*fs(:,i)',[],1);
    end
end
