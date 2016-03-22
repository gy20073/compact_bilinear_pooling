function [trainFs, testFs]=pool_rand_project_sparse(trainFs, testFs, outDim)
    tic
    fprintf('random projection sparse: ')
    a=cat(2, trainFs, testFs);
    nfeature=size(a, 1);
    
    r=rand(outDim, size(a,1), 'single');
    s=1.0/sqrt(nfeature);
    ind1=sparse(r>(1-0.5*s));
    ind_1=sparse(r<(0.5*s));
    mx=ind1-ind_1;
    a=single(mx*double(a));

    ntrain=size(trainFs, 2);
    trainFs=a(:, 1:ntrain);
    testFs=a(:, (ntrain+1):end);
    toc
end
