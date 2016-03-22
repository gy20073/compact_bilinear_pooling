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
        hs=randi(outDim, 2, c);
        ss=randi(2, 2, c)*2-3;
        
        comp=cell(1,2);
        ia=cell(1,2);
        for i=1:2
            [~, ia{i}, ~]=unique(hs(i,:)); % C=hi(ia), ia equals idim
            comp{i}=setdiff(1:numel(hs(i,:)), ia{i}); % the complement repeat ele
        end
        
        layer=struct('h',hs, 'weights', {{ss}}, ...
            'outDim', outDim, 'ia', {ia}, 'comp', {comp}, ...
            'learnW', true);
        
        X=rand(5, 5,c,3,'single')+1;
        dzdy=rand(5,5,outDim,3,'single')+1;    
    end
    
    fwd=@(X)getfield(yang_compact2_forward_noPool(...
                     layer, struct('x', X), struct()), 'x');
    
    dzdx=getfield(yang_compact2_backward_noPool(...
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
    dzdx=getfield(yang_compact2_backward_noPool(...
         layer, struct('x', X), struct('dzdx', dzdy)), 'dzdw');
    dzdx=dzdx{1};
     
    numerical_der_step=1e-2;
    fwd_w=@(w)getfield(yang_compact2_forward_noPool(...
                     struct('h', hs, 'weights', {{w}}, 'outDim',outDim, 'ia',{ia}, 'comp',{comp}, 'learnW', true),...
                     struct('x', X), struct()), 'x');
     [x, xp]=vl_testder(@(w)double(fwd_w(w)),  ...
                       ss,double(dzdy),double(dzdx),...
                       numerical_der_step);
    
    % analyze absolute error
    analyze_error(x-xp);
    % analyze relative error
    analyze_error((x-xp)./(x+1e-10));
end
