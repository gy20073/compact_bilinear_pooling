function yang_test_compact_bilinear(w1, w2, X, dzdy)
    function analyze_error(err)
        st=sort(abs(err(:)));
        max10=st(end-9:end);
        fprintf('mean error = %f. Top 10 errors: \n', mean(st));
        max10
    end
    
    if nargin<4
        w1=randi(2, 10, 30)*2-3;
        w2=randi(2, 10, 30)*2-3;

        X=rand(13,13,10,5,'single')+1;
        dzdy=rand(1,1,30,5,'single')+1;    
    end
    
    fwd=@(X)getfield(yang_compact_bilinear_RM_forward(...
                     struct('weights',{{w1,w2}}), struct('x', X), struct()), 'x');
    
    dzdx=getfield(yang_compact_bilinear_RM_backward(...
         struct('weights',{{w1,w2}}, 'learnW', false), struct('x', X), struct('dzdx', dzdy)), 'dzdx');
    numerical_der_step=1e-2;
    % output: my_derivative, numerical derivative
    [x, xp]=vl_testder(@(X)double(fwd(X)),  ...
                       double(X),double(dzdy),double(dzdx),...
                       numerical_der_step);
    
    % analyze absolute error
    analyze_error(x-xp);
    % analyze relative error
    analyze_error((x-xp)./(x+1e-10));
    
    %% test the derivative w.r.t. the weight parameters
    dzdx=getfield(yang_compact_bilinear_RM_backward(...
         struct('weights',{{w1,w2}}, 'learnW', true), struct('x', X), struct('dzdx', dzdy)), 'dzdw');
    dzdx=cat(1, dzdx{1}, dzdx{2});
    
    numerical_der_step=1e-2;
    fwd_w=@(w)getfield(yang_compact_bilinear_RM_forward(...
                     struct('weights',{{w(1:(end/2), :),w((end/2+1):end, :)}}), struct('x', X), struct()), 'x');
     [x, xp]=vl_testder(@(w)double(fwd_w(w)),  ...
                       double(cat(1, w1,w2)),double(dzdy),double(dzdx),...
                       numerical_der_step);
    
    % analyze absolute error
    analyze_error(x-xp);
    % analyze relative error
    analyze_error((x-xp)./(x+1e-10));
end
