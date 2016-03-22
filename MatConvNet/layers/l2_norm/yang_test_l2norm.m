
X=rand(1,1,30,40,'single');
fwd=@(X)getfield(yang_l2norm_forward('', struct('x', X), struct()), 'x');
Y=fwd(X);

dzdy=rand(1,1,30,40,'single');        
dzdx=getfield(yang_l2norm_backward('', struct('x', X), struct('dzdx', dzdy)), 'dzdx');
[x, xp]=vl_testder(fwd,X,dzdy,dzdx);

x=reshape(x, 1, []);
xp=reshape(xp, 1, []);
diff=abs(x-xp);
maxSev=sort(diff);
maxSev=maxSev(end-9:end);
fprintf('mean diff: %f \n', mean(diff));
maxSev
