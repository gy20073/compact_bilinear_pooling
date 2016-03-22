function yang_test_compact2_noPool(w1, w2, X, dzdy)
    function analyze_error(err)
        st=sort(abs(err(:)));
        max10=st(end-9:end);
        fprintf('mean error = %f. Top 10 errors: \n', mean(st));
        max10
    end
    
    if nargin<4
        outDim=30;
        c=10;
        
        obj=yang_compact_bilinear_TS_nopool_2stream(outDim, [c c], 1);
        w=cat(1, obj.weights_{:});
        w0=w;
        
        layer=struct('obj', obj, ...
            'outDim', outDim, ...
            'weights', {{ w }}, ...
            'learnW', true);
        
        X=rand(5, 5,c,3,'single')+1;
        dzdy=rand(5,5,outDim,3,'single')+1;    
    end
    
    fwd=@(X)getfield(obj.forward_simplenn_nopool(...
                     layer, struct('x', X), struct()), 'x');
    
    dzdx=getfield(obj.backward_simplenn_nopool(...
         layer, struct('x', X), struct('dzdx', dzdy)), 'dzdx');
    numerical_der_step=1e-3;
    % output: my_derivative, numerical derivative
    [x, xp]=vl_testder(@(X)double(fwd(X)),  ...
                       double(X),double(dzdy),double(dzdx),...
                       numerical_der_step);
    
    % analyze absolute error
    analyze_error(x-xp);
    % analyze relative error
    analyze_error((x-xp)./(x+1e-10));
    
    %% test the derivative w.r.t. the weight parameters
    dzdx=getfield(obj.backward_simplenn_nopool(...
         layer, struct('x', X), struct('dzdx', dzdy)), 'dzdw');
    dzdx=dzdx{1};
     
    numerical_der_step=1e-2;
    fwd_w=@(w)getfield(obj.forward_simplenn_nopool(...
                     struct('weights', {{w}}, 'learnW', true),...
                     struct('x', X), struct()), 'x');
     [x, xp]=vl_testder(@(w)double(fwd_w(w)),  ...
                       w0,double(dzdy),double(dzdx),...
                       numerical_der_step);
    
    % analyze absolute error
    analyze_error(x-xp);
    % analyze relative error
    analyze_error((x-xp)./(x+1e-10));
end
