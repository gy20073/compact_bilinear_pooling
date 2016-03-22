function yang_test_sqrt()
    function analyze_error(err)
        st=sort(abs(err(:)));
        max10=st(end-9:end);
        fprintf('mean error = %f. Top 10 errors: \n', mean(st));
        max10
    end

    X=rand(13,13,10,5,'single');
    fwd=@(X)getfield(yang_sqrt_forward('', struct('x', X), struct()), 'x');

    dzdy=rand(13,13,10,5,'single');        
    dzdx=getfield(yang_sqrt_backward('', struct('x', X), struct('dzdx', dzdy)), 'dzdx');
    numerical_der_step=1e-5;
    % output: my_derivative, numerical derivative
    [x, xp]=vl_testder(fwd,X,dzdy,dzdx, numerical_der_step);
    
    % analyze absolute error
    analyze_error(x-xp);
    % analyze relative error
    analyze_error((x-xp)./(x+1e-10));
end
