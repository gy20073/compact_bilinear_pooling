p=load('/Volumes/kraken.ist.berkeley.edu 2/local_data/fv_layer/exp01/mit-seed-01/dcnnfv-encoder.mat');

X=rand(2,3,512,2,'single');
clear yang_fv; 
Y=yang_fv(X, 'num_mixture', 64,...
             'samples_per_component', 1000,...
             'images_per_fv', 5000, ...
             'init_param', {p.means, p.covariances, p.priors});

dzdy=rand(1,1,65536,2,'single');        
dzdx=yang_fv(X, dzdy, Y);
[x, xp]=vl_testder(@yang_fv,X,dzdy,dzdx);

x=reshape(x, 1, []);
xp=reshape(xp, 1, []);
diff=abs(x-xp);
maxSev=sort(diff);
maxSev=maxSev(end-9:end);
fprintf('mean diff: %f \n', mean(diff));
maxSev
