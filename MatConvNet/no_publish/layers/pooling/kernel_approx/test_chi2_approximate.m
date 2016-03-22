sps=randsample(size(trainFV_l1, 2), 200);
t1=trainFV_l1(:, sps(1:100));
t2=trainFV_l1(:, sps(101:200));

% the approximation is pretty good
test_chi2_sub(t1, t2)
%%  test the approximation on the spatio pooling data
[trainFV_l1conv, valFV_l1conv]=normalize_l1_indi(trainFV, valFV);
%%
fs=trainFV_l1conv;
[h,w,c,n]=size(fs);
fs=permute(fs, [3,1,2,4]);
fs=reshape(fs, c, h*w*n);
%% test chi2
sps=randsample(size(fs, 2), 2000);
t1=fs(:, sps(1:1000));
t2=fs(:, sps(1001:2000));

%% the approximation of chi2 is pretty good
[real, approx]=test_chi2_sub(t1, t2);

%% test poly fit
fs=trainFV_poly;
testN=1000;

sps=randsample(size(fs, 2), testN);
t1=trainFV(:,:,:, sps(1:testN/2));
t2=trainFV(:,:,:, sps((testN/2+1):end));
p1=fs(:, sps(1:testN/2));
p2=fs(:, sps((testN/2+1):testN));

[real, approx]=test_chi2_sub(t1, t2, p1, p2, @dist_bilinear);