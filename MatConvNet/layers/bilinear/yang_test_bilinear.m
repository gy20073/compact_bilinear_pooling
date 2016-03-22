function yang_test_bilinear()
    function analyze_error(err)
        st=sort(abs(err(:)));
        max10=st(end-9:end);
        fprintf('mean error = %f. Top 10 errors: \n', mean(st));
        max10
    end

    X=rand(13,13,10,5,'single')+1;
    fwd=@(X)getfield(yang_bilinear_forward('', struct('x', X), struct()), 'x');

    dzdy=rand(1,1,100,5,'single')+1;        
    dzdx=getfield(yang_bilinear_backward('', struct('x', X), struct('dzdx', dzdy)), 'dzdx');
    numerical_der_step=1e-2;
    % output: my_derivative, numerical derivative
    [x, xp]=vl_testder(@(X)double(fwd(X)),  ...
                       double(X),double(dzdy),double(dzdx),...
                       numerical_der_step);
    
    % analyze absolute error
    analyze_error(x-xp);
    % analyze relative error
    analyze_error((x-xp)./(x+1e-10));
end
